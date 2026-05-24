# Sat2Street-DisasterGen

**Synthesizing Post-Disaster Street Views from Satellite Imagery via Generative Vision Models**

[![arXiv](https://img.shields.io/badge/arXiv-2603.20697-b31b1b.svg)](https://arxiv.org/abs/2603.20697)
[![IGARSS](https://img.shields.io/badge/Conference-IGARSS%202026-blue.svg)](https://igarss2026.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Accepted at **IEEE IGARSS 2026**

## Updates

- The IGARSS 2026 technical program scheduled our paper, “Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models,” as **Paper TH3.R6.4** in **Session TH3.R6**, with presentation time **Thursday, August 13, 2026, 14:30–14:45** in **Washington, D.C.**
- Yifan Yang received the **2026 IEEE GRSS Travel Grant** for **IGARSS 2026, Washington, D.C.** in the amount of **$500**.

---

## Overview

Ground-level street-view imagery is critical for post-disaster damage assessment, yet is often unavailable immediately after events due to access constraints. This project benchmarks whether **realistic and semantically consistent street views can be synthesized from satellite imagery**, and evaluates how different generative paradigms perform under this cross-view setting.

<p align="center">
  <img src="Figure/method.drawio.png" width="90%">
</p>

---

## Evaluated Methods

| Method | Paradigm | Key Characteristic |
|---|---|---|
| **Pix2Pix** | Conditional GAN | Preserves spatial layout; prone to texture blur |
| **SD 1.5 + ControlNet** | Diffusion | High realism; may underestimate fine-grained damage |
| **ControlNet + VLM (Gemini)** | Diffusion + Semantic | Damage-aware; possible semantic–geometry tension |
| **Disaster-MoE** | Mixture of Experts | Severity-adaptive; decouples structure from texture |

---

## Results

### Qualitative Comparison

![Qualitative Comparison](https://raw.githubusercontent.com/rayford295/Sat2Street-DisasterGen/main/Figure/compare%20result.PNG)

### Semantic Consistency (ResNet-18 Classifier)

![Confusion Matrices](https://raw.githubusercontent.com/rayford295/Sat2Street-DisasterGen/main/Figure/confusion_matrices_comparison.png)

- **SD 1.5 + ControlNet** achieves the highest semantic consistency (F1 ≈ 0.71)
- **Pix2Pix** exhibits strong mode collapse toward *mild* damage predictions
- **Gemini-guided and MoE** improve visual realism with a slight trade-off in class separability

---

## Citation

```bibtex
@article{yang2026satellite,
  title   = {Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite
             Imagery via Generative Vision Models},
  author  = {Yang, Yifan and Zou, Lei and Jepson, Wendy},
  journal = {arXiv preprint arXiv:2603.20697},
  year    = {2026}
}
```

---

## Acknowledgement

Supported by the Texas A&M University Environment and Sustainability Initiative (ESI) through the Environment and Sustainability Graduate Fellow Award.

---

## Contact

**Yifan Yang** — Department of Geography, Texas A&M University
[yyang295@tamu.edu](mailto:yyang295@tamu.edu) · [rayford295.github.io](https://rayford295.github.io)
