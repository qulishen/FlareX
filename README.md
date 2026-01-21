## FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering ✨

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/2510.09995)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green)](https://www.kaggle.com/datasets/lishenqu/flarex)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=qulishen.FlareX&right_color=violet)](https://github.com/qulishen/FlareX)
<!-- [![GitHub Stars](https://img.shields.io/github/stars/qulishen/FlareX?style=social)](https://github.com/qulishen/FlareX) -->

<p align="center">
  <img src="logo1.png" width="1000px">
</p>

This repository contains the official implementation of our NeurIPS 2025 paper 📄:
<p>
<div><strong>FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering</strong></div>
<div>
  <a href="https://qulishen.github.io/">Lishen Qu</a>,
  <a href="https://qulishen.github.io/">Zhihao Liu</a>,
  <a href="https://jspan.github.io/">Jinshan Pan</a>,
  <a href="https://joshyzhou.github.io/">Shihao Zhou</a>,
  <a href="https://jingleishi.github.io/">Jinglei Shi</a>,
  <a href="https://github.com/Calvin11311">Duosheng Chen</a>,
  <a href="https://cv.nankai.edu.cn/">Jufeng Yang</a>
</div>
<div>Accepted to <strong>NeurIPS 2025</strong> 🎉</div>

---

## TL;DR 🚀 | 快速概览

- **Train (random synthesis)**: `./dist_train.sh GPU_NUM options/random_dataloader/uformer_option.yml`
- **Train (depth-aware synthesis)**: `./dist_train.sh GPU_NUM options/deep_dataloader/uformer_option.yml` (requires `dataset/24K_res/`)
- **Evaluate / Inference**: `python test_evaluate.py --input ... --output ... --gt ...`

---

## Table of Contents 🧭 | 目录

- [Repository Structure](#repository-structure--仓库结构)
- [Installation](#installation--安装)
- [Data Preparation](#data-preparation--数据准备)
- [Optional: Depth Preprocessing (DPT)](#optional-depth-preprocessing-dpt--可选深度预处理)
- [Training](#training--训练)
- [Testing & Inference](#testing--inference--测试与推理)
- [Test Results](#test-results--结果)
- [Citation](#citation--引用)

---

## Repository Structure 📁 | 仓库结构

```text
FlareX/
├── basicsr/                         # Core training / models / datasets
├── options/                         # YAML configs (random_dataloader, deep_dataloader)
├── dist_train.sh                    # Distributed training entry (torchrun)
├── test_evaluate.py                 # Inference + metrics
├── requirements.txt                 # Python deps
└── dataset/                         # Put datasets here (see below)
```

---

## Installation 🛠️ | 安装

### 1) Create environment (recommended) 🐍

```bash
conda create -n flarex python=3.9 -y
conda activate flarex
```

### 2) Install dependencies 📦

```bash
pip install -r requirements.txt
```
---

## Data Preparation 🗂️ | 数据准备

This codebase expects the following paths by default (from `options/*.yml`) ✅

### 1) FlareX (Kaggle) ✨

Download from Kaggle:
- 👉 [FlareX Dataset](https://www.kaggle.com/datasets/lishenqu/flarex)

Place it under `dataset/FlareX/` and keep the internal structure, e.g.:

```text
dataset/
└── FlareX/
    ├── Flare2D/
    │   ├── input/                   # used by configs as scattering_dict
    │   └── gt/                      # used by configs as light_dict
    └── Flare3D/
        ├── input/                   # used by deep configs (image_pair_path.dataroot_lq)
        └── gt/                      # used by deep configs (image_pair_path.dataroot_gt)
```

### 2) Backgrounds: Flickr24K 🌄

Download from Google Drive:
- 👉 [Flickr24K Dataset](https://drive.google.com/file/d/1GNFGWfUbgXfELx5fZtjTjU2qqWnEa-Lr/view)

Place it under:

```text
dataset/
└── Flickr24K/                       # training background images
```

---

## Optional: Depth Preprocessing (DPT) 🔍 | 可选深度预处理

Depth-aware synthesis reads precomputed depth maps from `dataset/24K_res/` (see `basicsr/data/flareX_dataset.py`).

### 1) Get DPT

```bash
git clone https://github.com/isl-org/DPT.git
cd DPT
```

Download the DPT pretrained weights and place them in `DPT/weights/`.

### 2) Predict depth for Flickr24K

```bash
python run_monodepth.py -i ../dataset/Flickr24K -o ../dataset/24K_res
```

Expected output:

```text
dataset/
└── 24K_res/                         # depth maps (*.png) for Flickr24K
```

---

## Training 🏋️ | 训练

### A) Random synthesis (no depth) 🎲

```bash
./dist_train.sh GPU_NUM options/random_dataloader/uformer_option.yml
```

- `GPU_NUM`: number of GPUs (e.g., `2`)
- You can change model/config by swapping the YAML, e.g.:
  - `options/random_dataloader/{uformer,mprnet,restormer,hinet}_option.yml`

### B) Depth-aware synthesis 🧠

Make sure `dataset/24K_res/` is ready first, then run:

```bash
./dist_train.sh GPU_NUM options/deep_dataloader/uformer_option.yml
```

> Tip: if you hit port conflicts, set `PORT`:
>
> ```bash
> PORT=29501 ./dist_train.sh 2 options/deep_dataloader/uformer_option.yml
> ```

---

## Testing & Inference 🔎 | 测试与推理

`test_evaluate.py` supports inference + metric computation (PSNR/SSIM/LPIPS) 🧾

### Example

```bash
python test_evaluate.py \
  --input path/to/input_images \
  --output path/to/output_dir \
  --model_type Uformer \
  --model_path weight/Uformer/Uformer_2d_3d_deep_1800.pth \
  --gt path/to/gt_images
```

Optional:
- `--mask path/to/masks` (if you have masks; used for extra metrics)

---

## Test Results 📊 | 结果

Pretrained results and files:
- [Google Drive](https://drive.google.com/drive/folders/14BrL5oNOWcd5WZMfjkpLg7js4yjnhY7P?usp=sharing)

---

## Citation 📚 | 引用

If you find this work useful, please cite:

```bibtex
@inproceedings{FlareX_lishenqu,
    title={FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering},
    author={Lishen, Qu and Zhihao, Liu and Jinshan, Pan and Shihao, Zhou and Jinglei, Shi and Duosheng, Chen and Jufeng, Yang},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```
