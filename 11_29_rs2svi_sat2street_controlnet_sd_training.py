import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm

# ==========================================
# 0. 显卡自检 (Pre-flight Check)
# ==========================================
print(f"🔍 正在检查环境...")
if torch.cuda.is_available():
    print(f"✅ 显卡已就绪: {torch.cuda.get_device_name(0)}")
    print(f"🚀 PyTorch 版本: {torch.__version__}")
else:
    raise RuntimeError("❌ 致命错误：依然没检测到显卡！请务必先【重启内核】再运行！")

# ==========================================
# 1. 配置区域
# ==========================================
BATCH_SIZE = 6          # 3090 24G 显存，跑 6 张图没问题
NUM_EPOCHS = 10         # 训练 10 轮
LEARNING_RATE = 1e-5    # 学习率
OUTPUT_DIR = "./sat2street_output" # 模型保存路径 (会保存在当前文件夹下)
MODEL_ID = "runwayml/stable-diffusion-v1-5" # 基础模型
VAL_RATIO = 0.1         # 10% 的数据留作测试集

# ==========================================
# 2. 智能数据集类 (带自动划分功能)
# ==========================================
class Sat2StreetDataset(Dataset):
    def __init__(self, root_dir="./", resolution=512, split="train", val_ratio=0.1):
        """
        root_dir: 当前目录 (因为你的 notebook 就在 data 文件夹里)
        split: 'train' (训练集) 或 'val' (验证/测试集)
        """
        self.resolution = resolution
        self.root_dir = root_dir
        
        # 1. 寻找 csv
        csv_path = os.path.join(root_dir, "pairs.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到 {csv_path}！请确认文件就在旁边。")

        # 2. 读取并打乱数据
        df_all = pd.read_csv(csv_path)
        # 固定随机种子 42，确保每次划分都一样
        df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 3. 划分训练集和验证集
        val_size = int(len(df_all) * val_ratio)
        train_size = len(df_all) - val_size
        
        if split == "train":
            self.df = df_all.iloc[:train_size] # 取前 90%
            print(f"✅ [{split.upper()}] 加载成功: {len(self.df)} 张图片 (训练用)")
        else:
            self.df = df_all.iloc[train_size:] # 取后 10%
            print(f"✅ [{split.upper()}] 加载成功: {len(self.df)} 张图片 (测试用)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 无论 CSV 里写啥路径，只取文件名
        sat_name = os.path.basename(row["sat_path"])
        svi_name = os.path.basename(row["svi_path"])
        
        # 拼接本地真实路径
        sat_path = os.path.join(self.root_dir, "images", sat_name)
        svi_path = os.path.join(self.root_dir, "images", svi_name)

        # 读取与处理
        try:
            sat = Image.open(sat_path).convert("RGB").resize((self.resolution, self.resolution))
            svi = Image.open(svi_path).convert("RGB").resize((self.resolution, self.resolution))
        except FileNotFoundError:
            # 容错处理：如果找不到图，打印一下
            print(f"⚠️ 找不到图片: {sat_path}")
            # 返回一个黑图防止崩溃 (或者你可以选择报错)
            sat = Image.new('RGB', (self.resolution, self.resolution))
            svi = Image.new('RGB', (self.resolution, self.resolution))

        svi_t = torch.from_numpy(np.array(svi).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        sat_t = torch.from_numpy(np.array(sat).astype(np.float32) / 255.0).permute(2, 0, 1)

        prompt = f"street view photography, realistic, ground level view, {str(row['severity']).replace('_', ' ').lower()}, high quality, 4k"

        return {"pixel_values": svi_t, "condition_pixel_values": sat_t, "input_ids": prompt}

# ==========================================
# 3. 训练主程序
# ==========================================
def train_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16")
    
    # ---------------------------------------------------------
    # 加载数据 (Train / Val)
    # ---------------------------------------------------------
    try:
        # 只训练 90% 的数据
        train_dataset = Sat2StreetDataset(split="train", val_ratio=VAL_RATIO)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 加载模型
    print("🚀 正在初始化模型...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    # 冻结参数
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    def collate_fn(examples):
        pixel_values = torch.stack([x["pixel_values"] for x in examples])
        condition = torch.stack([x["condition_pixel_values"] for x in examples])
        prompts = [x["input_ids"] for x in examples]
        inputs = tokenizer(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        return {"pixel_values": pixel_values, "condition_pixel_values": condition, "input_ids": inputs.input_ids}

    # 🔥 Windows 关键设置: num_workers=0 🔥
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)

    controlnet, optimizer, train_dataloader = accelerator.prepare(controlnet, optimizer, train_dataloader)
    
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    print("🔥 训练开始！")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                down, mid = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch["condition_pixel_values"].to(dtype=torch.float16),
                    return_dict=False
                )
                
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=torch.float16) for s in down],
                    mid_block_additional_residual=mid.to(dtype=torch.float16)
                ).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        accelerator.unwrap_model(controlnet).save_pretrained(save_path)
        print(f"💾 模型已保存: {save_path}")

if __name__ == "__main__":
    train_main()
