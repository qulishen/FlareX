# FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/xxxx.xxxxx)  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green)](https://www.kaggle.com/datasets/lishenqu/flarex)

This repository contains the official implementation of our NeurIPS 2025 paper:  
**FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering**.  

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

## Testing & Inference

Evaluate a trained model with:

```bash
python test_evaluate.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{qu2025flarex,
  title={FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering},
  author={Lishen Qu, Zhihao Liu, Jinshan Pan, Shihao Zhou, Jinglei Shi, Duosheng Chen, Jufeng Yang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
---
