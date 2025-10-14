# FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/2510.09995)  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green)](https://www.kaggle.com/datasets/lishenqu/flarex)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=qulishen.FlareX&right_color=violet)](https://github.com/qulishen/FlareX)
<!-- [![GitHub Stars](https://img.shields.io/github/stars/qulishen/FlareX?style=social)](https://github.com/qulishen/FlareX) -->

<p align="center">
  <img src="logo1.png" width="1000px"> 
</p>

This repository contains the official implementation of our NeurIPS 2025 paper:  
<p>
<div><strong>FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering</strong></div>
<div><a href="https://qulishen.github.io/">Lishen Qu</a>, 
   	<a href="https://qulishen.github.io/">Zhihao Liu</a>,
    <a href="https://jspan.github.io/">Jinshan Pan</a>, 
    <a href="https://joshyzhou.github.io/">Shihao Zhou</a>,
    <a href="https://jingleishi.github.io/">Jinglei Shi</a>,
    <a href="https://github.com/Calvin11311">Duosheng Chen</a>,
    <a href="https://www4.comp.polyu.edu.hk/~cslzhang/">Lei Zhang</a>,
    <a href="https://cv.nankai.edu.cn/">Jufeng Yang</a>
    </div>
<div>Accepted to <strong>NeurIPS 2025</strong></div>

---

## Repository Structure

```

FlareX/
├── requirements.txt       # Dependencies
├── dist_train.sh          # Distributed training script
├── test_evaluate.py       # Testing and evaluation
└── dataset

````

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/FlareX.git
cd FlareX
pip install -r requirements.txt
````

---

## Dataset

Download the dataset from Kaggle:
👉 [FlareX Dataset](https://www.kaggle.com/datasets/lishenqu/flarex)

Place the dataset under the `dataset/` directory:

## Training

Run distributed training with:

```bash
./dist_train.sh GPU_NUM option
```

* `GPU_NUM`: number of GPUs to use (e.g., 2)
* `option`: additional arguments (e.g., config file)

---

## Pretrained Model

[Google Drive](https://drive.google.com/file/d/1oILbfk3ZZt_uctp1cY11Km9fWKsy0rAW/view?usp=sharing)

## Testing & Inference

Evaluate a trained model with:

```bash
python test_evaluate.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{FlareX_lishenqu,
    title={FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering},
    author={Lishen, Qu and Zhihao, Liu and Jinshan, Pan and Shihao, Zhou and Jinglei, Shi and Duosheng, Chen and Jufeng, Yang},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```
---
