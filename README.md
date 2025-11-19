# RSDN-GZSL: Recurrent Semantic Disentangling Network for Generalized Zero-Shot Learning
A PyTorch implementation of the paper "Recurrent Semantic Disentangling Network for Generalized Zero-Shot Learning", which achieves state-of-the-art performance on three mainstream ZSL benchmarks (AWA2, SUN, CUB).

## 🌟 Core Highlights
- **Paradigm Shift**: Replaces static class-level semantics with dynamic instance-specific semantics, solving the misalignment between visual features and predefined semantics.
- **Closed-Loop Iteration**: Builds a cyclic mechanism between feature disentanglement and semantic refinement to progressively enhance discriminative feature extraction.
- **Dual-Module Synergy**: Integrates Dual-Path Feature Generation Network (DPFGN) for semantic-visual alignment and Evolution Attention Network (EAN) for class consistency preservation.
- **Strong Generalization**: Outperforms SOTA methods on both fine-grained (CUB, SUN) and coarse-grained (AWA2) datasets, and is compatible with multiple generative ZSL frameworks.

<img width="1400" height="700" alt="image" src="https://github.com/user-attachments/assets/1d8942fe-08d2-4a03-b994-40b3b86e76d9" />



## 📋 Overview
Generalized Zero-Shot Learning (GZSL) aims to classify both seen and unseen classes without training data for unseen categories. Traditional methods rely on static expert-defined semantics, leading to overemphasis on high-frequency attributes and neglect of sample-specific features—especially in fine-grained scenarios.

RSDN-GZSL addresses this limitation by:
1. Generating dynamic instance-level semantics via DPFGN (sample path for disentanglement, class path for visual prototypes).
2. Refining semantics with EAN to ensure class consistency, using visual prototypes as anchors.
3. Feeding refined semantics back to DPFGN for iterative feature enhancement, forming a closed loop.

<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/ecf0dfd7-04e8-4d55-ba40-6f2f465db09d" />



## 🚀 Quick Start
### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/JingHu-gdut/RSDN-GZSL.git
cd RSDN-GZSL

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the three benchmark datasets and organize them as follows:
```
data/
├── AWA2/          # Animals with Attributes 2
├── SUN/           # SUN Attribute Dataset
└── CUB/           # Caltech UCSD Birds 200-2011
```
- **AWA2**: 50 animal classes (40 seen, 10 unseen), 85 attributes
- **SUN**: 717 scene classes (645 seen, 72 unseen), 102 attributes
- **CUB**: 200 bird classes (150 seen, 50 unseen), 1024 attributes

### 3. Training & Inference
#### Train RSDN-GZSL
```bash
# Train on CUB dataset (GZSL setting)
python train.py --dataset CUB --gpu 0 --epochs 300 --batch_size 64

# Train on AWA2 (CZSL setting)
python train.py --dataset AWA2 --setting CZSL --gpu 0 --epochs 150 --batch_size 64
```

#### Key Training Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (AWA2/SUN/CUB) | CUB |
| `--setting` | Task setting (CZSL/GZSL) | GZSL |
| `--gpu` | GPU device ID | 0 |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Batch size | 64 (AWA2/CUB), 128 (SUN) |
| `--lambda_cl` | Weight for contrastive loss | 0.5 |
| `--lambda_ce` | Weight for cross-entropy loss | 1.0 |

## 📊 Experimental Results
### State-of-the-Art Performance
| Dataset | Setting | Metric | RSDN-GZSL | SOTA (AENet/PSVMA) |
|---------|---------|--------|-----------|-------------------|
| AWA2 | CZSL | Acc | 74.7% | 74.9% |
| AWA2 | GZSL | H | 77.1% | 74.9% |
| SUN | CZSL | Acc | 77.1% | 70.4% |
| SUN | GZSL | H | 56.0% | 51.0% |
| CUB | CZSL | Acc | 86.0% | 80.3% |
| CUB | GZSL | H | 74.8% | 74.7% |

### Ablation Study
All components of RSDN-GZSL contribute significantly to performance (taking CUB as example):
- Baseline H-score: 66.8%
- Full model H-score: 74.8% (↑8.0%)
- Removal of FAN/RN/TC leads to 2-5% performance drop.

### Compatibility with Generative Frameworks
RSDN-GZSL can enhance existing generative ZSL models:
| Base Model | Dataset | H-score (Before) | H-score (After) | Improvement |
|------------|---------|------------------|-----------------|-------------|
| CLSWGAN | CUB | 65.6% | 73.4% | +7.8% |
| FREE | SUN | 52.4% | 59.4% | +7.0% |
| SHIP | AWA2 | 74.7% | 79.1% | +4.4% |

## 📝 Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{RSDNGZSL2025,
  title={Recurrent Semantic Disentangling Network for Generalized Zero-Shot Learning},
  author={Jing Hu et al.},
  journal={Neurocomputing},
  year={2025},
  note={Preprint submitted to Neurocomputing}
}
```

## 📄 References

We thank the following repos for providing helpful components in our work: <br>
-[FREE](https://github.com/shiming-chen/FREE) <br>
-[SHIP](https://github.com/mrflogs/SHIP)


