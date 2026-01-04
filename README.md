# Image Captioning with Hybrid LSTM-Transformer

> **Deep Learning Project**: End-to-end image captioning system using InceptionV3 feature extraction and hybrid LSTM-Transformer decoder architectures, trained on the MS COCO dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Project Overview](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-project-overview)
- [Architecture](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#%EF%B8%8F-architecture)
- [Dataset](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-dataset)
- [Installation](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-installation)
- [Usage](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-usage)
- [Results](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-results)
- [Project Structure](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-project-structure)
- [Key-Notebooks](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4?tab=readme-ov-file#-key-notebooks)

## 🎯 Project Overview

This project implements multiple image captioning architectures and compares their performance:- **Hybrid LSTM-Transformer**: Combines LSTM for sequential processing with Transformer cross-attention
- **Full Transformer Decoder**: Complete multi-layer transformer decoder with positional encoding
- **Simple Transformer**: Lightweight transformer for faster training

**Key Features:**
- Pre-extracted InceptionV3features (2048-dim) for efficient training
- Multiple model architectures for comparison
- LayerCAM visualizations for model interpretability
- BLEU score evaluation
- Comparison with state-of-the-art models (BLIP, ViT-GPT2, etc.)

## 🏗️ Architecture

### Hybrid LSTM-Transformer Model

```
Image (InceptionV3 features: 2048-dim)
    ↓
Caption Tokens → Embedding (256-dim)
    ↓
LSTM Layer (256 units, return_sequences=True)
    ↓
Transformer Cross-Attention (4 heads)
    ↓
Combine (Add + LayerNorm + Dropout)
    ↓
Dense Output (vocab_size=29,854)
```

**Model Specifications:**
- **Vocabulary Size**: 29,854 tokens
- **Max Caption Length**: 59 tokens
- **Feature Dimension**: 2,048 (InceptionV3)
- **Embedding Dimension**: 256
- **LSTM Units**: 256
- **Attention Heads**: 4
- **Dropout**: 0.2

## 📊 Dataset

**MS COCO 2017 Dataset** + Pre-extracted Features

### Raw Dataset
- **Source**: [COCO 2017 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- **Training Images**: 118,287
- **Validation Images**: 5,000
- **Total Captions**: ~616,000

### Kaggle Input Datasets

The project uses three Kaggle datasets:

1. **[COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)** - Raw images and annotations
   ```
   /kaggle/input/coco-2017-dataset/coco2017/
   ├── annotations/        # COCO annotation files
   ├── train2017/         # 118k training images
   ├── val2017/           # 5k validation images
   └── test2017/          # Test images
   ```

2. **mile-stone-1** - InceptionV3 Features (2048-dim)
   ```
   /kaggle/input/mile-stone-1/
   ├── train_features.pkl  # Training image features (118,287 × 2048)
   └── val_features.pkl    # Validation image features (5,000 × 2048)
   ```

3. **img-cap-v1** - Preprocessed Data & VGG16 Features
   ```
   /kaggle/input/img-cap-v1/
   ├── image_features.npy      # VGG16 features (616,767 × 512)
   ├── padded_sequences.npy    # Tokenized captions (616,767 × 59)
   ├── idx_to_word.pkl         # Vocabulary mapping
   └── metadata.pkl            # Dataset metadata
   ```

**Vocabulary**: 29,854 unique tokens  
**Max Caption Length**: 59 tokens  
**Feature Dimensions**: InceptionV3 (2048), VGG16 (512)

## 💻 Installation

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- CUDA (optional, for GPU support)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4.git
cd DEPI-ML-Project-G4
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add Kaggle datasets to your notebook:**
   
   If running on Kaggle, add these datasets as inputs:
   - [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
   - [mile-stone-1](https://www.kaggle.com/code/abdelwhabmohamed05/mile-stone-1)
   - [img-cap-v1](https://www.kaggle.com/code/usuffff/img-cap-v1/)
   
   If running locally, download the features and update paths in [`src/config.py`](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4/blob/main/src/config.py)

## 🚀 Usage

### Training

Train a model using the default configuration:

```bash
python scripts/train.py
```

Train a specific model variant:

```bash
# Hybrid LSTM-Transformer
EXPERIMENT_NAME=hybrid_inception_v3 python scripts/train.py

# Full Transformer
EXPERIMENT_NAME=transformer_inception_v3 python scripts/train.py

# Simple Transformer
EXPERIMENT_NAME=simple_transformer_inception_v3 python scripts/train.py
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py
```

### LayerCAM Visualization

Generate class activation maps for InceptionV3:

```bash
python scripts/LayerCAM.py path/to/your/image.jpg
```

### Model Comparison

Compare different models:

```bash
python scripts/model_comparison.py
```

## 📈 Results

We deployed our model in addition to Blib2, and VIT-GPT2 using azure: [deployment](https://imagecaption-d0gje9h2eba2frh2.francecentral-01.azurewebsites.net/)

### BLEU Scores

| Model | BLEU Score |
|-------|-----------|
| **Hybrid LSTM-Transformer (Ours)** | **7.85** |
| Simple Transformer (Ours) | 7.10 |
| Full Transformer (Ours) | 8.20 |
| BLIP | 22.50 |
| ViT-GPT2 | 18.30 |
| BLIP2 | 28.70 |
| GIT-Large | 25.40 |

### Performance Analysis

Our model achieves a BLEU score of ~8, which is modest compared to state-of-the-art models. This gap is expected as:
- We use frozen InceptionV3 features instead of fine-tuned vision encoders
- Limited model capacity (256-dim embeddings vs. 768+ in SOTA models)
- Greedy decoding vs. beam search
- Shorter training duration

**Potential Improvements:**
1. Fine-tune the vision encoder
2. Use larger pretrained models (ViT-Large)
3. Implement beam search decoding
4. Increase training epochs and model capacity
5. Apply advanced regularization techniques

## 📁 Project Structure

```
Image_Captioning/
├── src/                          # Source code package
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Configuration & hyperparameters
│   ├── model.py                 # Model architectures
│   ├── data.py                  # Data loading utilities
│   ├── dataset.py               # Dataset & data generators
│   ├── engine.py                # Training engine
│   └── utils.py                 # Helper functions
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── LayerCAM.py             # Visualization script
│   └── model_comparison.py      # Model comparison
│
├── models/                       # Saved models (gitignored)
├── records/                      # Training logs (gitignored)
├── plots/                        # Plots & visualizations
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🔬 Key Notebooks

Our development process is documented in these Kaggle notebooks:

1. **[Milestone 1: Data Preparation & EDA](https://www.kaggle.com/code/abdelwhabmohamed05/mile-stone-1)**
   - Feature extraction with InceptionV3
   - Vocabulary building
   - Data exploration

2. **[Model Training](https://www.kaggle.com/code/hobasaad05/model-train)**
   - Hybrid LSTM-Transformer implementation
   - Training loops and callbacks
   - Model evaluation

3. **[LayerCAM Visualization](https://www.kaggle.com/code/abdelwhabmohamed05/layercam-inception-v3/notebook)**
   - Class activation mapping
   - Model interpretability

4. **[Model Comparison](https://www.kaggle.com/code/nadenkhaled/image-captioning-model-comparison)**
   - Benchmarking against SOTA models
   - BLEU score analysis

## 👥 Contributors

- **Team Members :**
- **Abdelwhab Mohammed** - [GitHub](https://github.com/Abdelwhabmohammed)
- **Yousef Abdulati**
- **Nadine Khaled**  
- **Shahd Ezzat** 
- **Nouran Ahmed** 
- **Mahmoud Saad**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MS COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **InceptionV3**: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision"
- **Transformer Architecture**: Vaswani et al., "Attention Is All You Need"
- **LayerCAM**: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps"

## 📧 Contact

For questions or collaboration:
- Email: [abdelwhabmohammed0@gmail.com]
- Project Link: [https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4](https://github.com/Abdelwhabmohammed/DEPI-ML-Project-G4)

---

**Note**: This is a learning project developed as part of the DEPI Machine Learning track. The focus is on understanding and implementing image captioning architectures rather than achieving state-of-the-art performance.
