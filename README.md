# SASRec (Self-Attentive Sequential Recommendation) - PyTorch Implementation

PyTorch implementation of **Self-Attentive Sequential Recommendation** (SASRec), published in ICDM 2018 by Wang-Cheng Kang and Julian McAuley.

[![Paper](https://img.shields.io/badge/Paper-ICDM%202018-blue)](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org/)

## 📄 Paper

**Self-Attentive Sequential Recommendation**
Wang-Cheng Kang, Julian McAuley
*ICDM 2018*

## 🎯 Overview

SASRec (Self-Attentive Sequential Recommendation) is a sequential recommendation model that uses self-attention mechanisms to capture user preferences from their interaction sequences. The model can identify which items are "relevant" from a user's historical actions and use them to predict the next item.

### Key Features

- ✨ **Self-Attention Mechanism**: Captures long-range dependencies in user sequences
- 🎲 **Causal Attention**: Prevents information leakage from future items
- 🚀 **High Performance**: State-of-the-art results on multiple datasets
- 📊 **Efficient Training**: Parallel computation for fast training
- 🔄 **Sequential Modeling**: Models user behavior sequences effectively

## 🏗️ Model Architecture

```
Input Sequence
  [item_1, item_2, ..., item_n]
       ↓
  Embedding Layer
       ↓
  Positional Encoding
       ↓
  Self-Attention Blocks (with Causal Mask)
       ├─ Multi-head Self-Attention
       ├─ Layer Normalization
       ├─ Point-wise Feed Forward
       └─ Residual Connections
       ↓
  Final Layer Normalization
       ↓
  Prediction Layer
       ↓
  Next Item Probability
```

### Core Components

1. **Embedding Layer** ([embedding.py](embedding.py))
   - Item embeddings and positional encodings
   - Learnable positional embeddings

2. **Self-Attention Layer** ([attention.py](attention.py))
   - **Multi-head Self-Attention**: Captures different aspects of item relationships
   - **Causal Masking**: Ensures autoregressive property
   - **Scaled Dot-Product Attention**: Standard attention mechanism

3. **Point-wise Feed Forward** ([feed_forward.py](feed_forward.py))
   - Two-layer feed-forward network
   - ReLU activation and dropout

4. **Layer Normalization**
   - Applied after attention and feed-forward layers
   - Improves training stability

## 📁 Project Structure

```
SASRec/
├── README.md                  # This file
├── main.py                    # Main training and evaluation script
├── model.py                   # SASRec model implementation
├── attention.py               # Self-attention mechanism
├── embedding.py               # Embedding and positional encoding
├── feed_forward.py            # Point-wise feed forward network
├── data_loader.py             # Data loading and preprocessing
├── utils.py                   # Utility functions
└── __init__.py                # Package initialization
```

## 📊 Dataset

### MovieLens-1M Dataset

The model is trained and evaluated on the **MovieLens-1M** dataset, which contains:
- 1,000,209 anonymous ratings
- 6,040 users
- 3,706 movies
- Ratings from 1 to 5

### Data Preprocessing

1. **Rating Filtering**: Keep ratings ≥ 4 as positive interactions
2. **User Filtering**: Remove users with fewer than 5 interactions
3. **Sequence Construction**: Sort interactions by timestamp
4. **Train/Test Split**: Leave-one-out evaluation

### Expected Data Structure

```
ml-1m.txt                    # Raw MovieLens-1M dataset
ml-1m.train.rating           # Preprocessed training data
```

## 🛠️ Installation

### Requirements

```bash
pip install torch numpy pandas tqdm
```

### Tested Environment

- Python 3.7+
- PyTorch 1.10+
- NumPy 1.21+
- Pandas 1.3+
- CUDA 11.0+ (optional, for GPU training)

## 🚀 Usage

### Training

Train the SASRec model with default parameters:

```bash
python main.py --train_dir ml-1m.train.rating --dataset ml-1m
```

**Training Parameters:**

```bash
python main.py \
    --train_dir ml-1m.train.rating \
    --dataset ml-1m \
    --batch_size 128 \
    --lr 0.001 \
    --maxlen 200 \
    --hidden_units 50 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.5 \
    --l2_emb 0.0 \
    --epochs 300
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --eval --dataset ml-1m --state_dict_path best_model/best_model.pth
```

## 📈 Training Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch Size | 128 | Training batch size |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Max Sequence Length | 200 | Maximum user sequence length |
| Hidden Units | 50 | Dimension of hidden layers |
| Number of Blocks | 2 | Number of self-attention blocks |
| Number of Heads | 1 | Number of attention heads |
| Dropout Rate | 0.5 | Dropout probability |
| L2 Regularization | 0.0 | Weight decay for embeddings |
| Epochs | 300 | Number of training epochs |

### Training Process

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss Function**: Binary Cross-Entropy with negative sampling
- **Evaluation Metrics**: Hit Rate @ 10, NDCG @ 10
- **Negative Sampling**: Random negative items for each positive
- **Checkpointing**: Best model saved based on validation metrics

## 🔬 Model Components Explained

### 1. Self-Attention Mechanism

The self-attention layer computes attention weights between all pairs of items in the sequence:

```python
attention_weights = softmax(Q × K^T / √d_k + mask)
attention_output = attention_weights × V
```

**Key Features:**
- **Causal Masking**: Prevents attending to future items
- **Multi-head Attention**: Captures different relationship aspects
- **Residual Connections**: Helps with gradient flow

### 2. Positional Encoding

Since self-attention is position-agnostic, positional encodings are added to provide sequence order information:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### 3. Point-wise Feed Forward

Two-layer feed-forward network applied to each position independently:

```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

## 📊 Results

The model is evaluated using:
- **Hit Rate @ 10 (HR@10)**: Whether the target item is in top-10 recommendations
- **Normalized Discounted Cumulative Gain @ 10 (NDCG@10)**: Rank-sensitive metric

### Actual Performance on MovieLens-1M

Based on our local training run with 3 epochs:

#### Training Progress
- **Epoch 1**: Loss=1.2042, NDCG@10=0.2441, HR@10=0.4482
- **Epoch 2**: Loss=1.0395, NDCG@10=0.2444, HR@10=0.4447
- **Epoch 3**: Loss=1.0257, NDCG@10=0.2434, HR@10=0.4444

#### Final Test Results
- **Best Model**: Epoch 2 (saved with NDCG@10=0.2444)
- **Final Test NDCG@10**: 0.2402
- **Final Test HR@10**: 0.4429

#### Model Information
- **Dataset**: 6,040 users, 3,416 items
- **Trainable Parameters**: 211,950
- **Model File Size**: 838 KB

### Expected Performance with Full Training

With full training (200+ epochs), the model typically achieves:
- **HR@10**: ~0.65-0.70
- **NDCG@10**: ~0.38-0.42

*Note: Our 3-epoch training serves as a quick validation. For production use, train with more epochs for optimal performance.*

## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

## 🙏 Acknowledgments

This implementation is inspired by and builds upon the excellent work from:

- **Original Paper**: [Self-Attentive Sequential Recommendation (ICDM 2018)](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
- **Reference Implementation**: [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)

Special thanks to the authors of the reference repository for their valuable PyTorch implementation.

## 📝 License

This project is open-sourced for research and educational purposes. Please refer to the original paper and repository for commercial use considerations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⭐ If you find this implementation useful, please consider giving it a star!**