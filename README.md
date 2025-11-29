# RS2SV: Remote Sensing → Street‑View Generation

**Goal**: Research code for generating plausible street‑view images conditioned on overhead/remote‑sensing inputs, with a focus on disaster geography and cross‑view consistency.

## Features (planned)
- Data pipeline for pairing overhead tiles with street‑view images (Mapillary-recommended).
- Config‑driven experiments (Hydra/OmegaConf).
- Baseline models: conditional diffusion (UNet), cross‑view encoder, super‑resolution for details.
- Evaluation: FID/KID, CLIPScore, Geo‑consistency probes, human study templates.
- Reproducible runs + experiment tracking (W&B or TensorBoard).

## Quickstart
```bash
# 1) Create environment
conda env create -f environment.yml
conda activate rs2sv

# 2) (Optional) set up DVC for data versioning
# dvc init
# dvc remote add -d storage <your-remote>
# dvc pull

# 3) Smoke test (imports + style)
make ci

# 4) Train baseline (example)
python -m src.train experiment=default
```

> **Data**: Use Mapillary (open license) or your own datasets. Do **NOT** upload proprietary/terms-restricted data to the repo. Large files are tracked via Git LFS patterns in `.gitattributes` (see below).

## Repo Layout
```
.
├── configs/                # Hydra configs
│   └── default.yaml
├── src/
│   ├── data/               # datamodules, dataset readers
│   │   └── datamodule.py
│   ├── models/             # diffusion / cross-view modules
│   │   └── baseline.py
│   ├── utils/              # geo utils, logging, metrics
│   │   └── geo.py
│   └── train.py            # entry point
├── notebooks/              # experiments & visualization
├── experiments/            # logs, checkpoints (gitignored)
├── tests/                  # minimal tests
├── environment.yml
├── .gitignore
├── .gitattributes          # LFS for large binaries
├── LICENSE
├── Makefile
└── .github/workflows/ci.yml
```

## Ethics & Responsible AI
- Respect platform terms (e.g., Mapillary) and privacy. Never attempt to re‑identify people/plates/faces.
- Include disclaimers when generating disaster‑area views: outputs are **synthetic** and may be wrong.
- Log model/data versions and seed for reproducibility.

## Citation
If you use this repo, please cite:
```
Yang, Y. (2025). RS2SV: Remote Sensing to Street‑View Generation (v0.1). GitHub repository.
```
