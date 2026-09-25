# Vision Transformer on CIFAR-10

## Overview

This project implements a **Vision Transformer (ViT) from scratch using PyTorch** for 10-class image classification on the CIFAR-10 dataset.

The implementation includes patch embedding, learnable positional embeddings, a learnable CLS token, multi-head self-attention, MLP blocks, residual connections, layer normalization, dropout, and a classification head.

No pre-trained model or pre-trained weights are used.

## Notebook

The complete implementation is provided in:

`notebook9acefc3696_(3)_colliisc (2).ipynb`

The notebook contains data loading, preprocessing, train/validation splitting, model implementation, training, validation, best-model selection, and final evaluation on the official CIFAR-10 test set.

## How to Run in Google Colab

1. Open `notebook9acefc3696_(3)_colliisc (2).ipynb` in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select a **GPU** runtime.
4. The experiment was run using a **T4 GPU**.
5. Run the notebook cells from top to bottom.
6. The notebook mounts Google Drive to access the CIFAR-10 dataset.
7. The notebook trains the Vision Transformer and evaluates the selected best model on the official CIFAR-10 test set.

## Dataset

The model is trained and evaluated on **CIFAR-10**, containing 10 image classes.

* Training images: 50,000
* Test images: 10,000
* Image size: 32 × 32
* Number of classes: 10
* Training split: 45,000 images
* Validation split: 5,000 images

The 5,000-image validation set is created from the original CIFAR-10 training set. The official CIFAR-10 test set is reserved for final evaluation.

## Model Configuration

 Parameter                                        Value 
============                                     =======
 Image size                                       32 × 32 
 Patch size                                         4 × 4 
 Number of patches                                     64 
 Embedding dimension                                  256 
 Transformer blocks                                     6 
 Attention heads                                        8 
 MLP dimension                                        512 
 Dropout                                              0.1 
 Number of classes                                     10 
 CLS token                                      Learnable 
 Positional embedding                           Learnable 
 Attention               Global multi-head self-attention 

## Training Configuration

  Setting                                           Value 
 =============                                     =======
 Optimizer                                           Adam 
 Learning rate                                       3e-4 
 Batch size                                           128 
 Epochs                                                45 
 Loss function                         Cross-Entropy Loss 
 Random seed                                           42 
 GPU                                                   T4 
 Training split                             45,000 images 
 Validation split                            5,000 images 
 Model selection                 Best validation accuracy 
 Approximate training runtime               40–50 minutes 

## Implementation Details

The Vision Transformer components are implemented directly in PyTorch rather than using a pre-built Vision Transformer or Transformer encoder.

The implementation includes:

* Patch embedding using convolution
* Learnable CLS token
* Learnable positional embeddings
* Query (Q), Key (K), and Value (V) projections
* Scaled dot-product attention
* Softmax attention weights
* Multi-head attention
* Head merging
* MLP block with GELU activation
* Residual connections
* Custom layer normalization
* Dropout
* Final classification head

### Scaled Dot-Product Attention

The attention mechanism computes:

**Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V**

The notebook explicitly computes the query-key dot product, scales the attention scores by the square root of the head dimension, applies softmax, and uses the resulting attention weights to combine the value vectors.

### Layer Normalization

Layer normalization is implemented manually rather than using a pre-built `nn.LayerNorm` module.

The implementation computes the mean and variance and applies learnable scale and shift parameters.

## Results

The model achieved the following final test performance:

 Model          Test Top-1 Accuracy  

 Standard ViT            **74.75%** 

The final test evaluation was performed on the official CIFAR-10 test set after selecting the model based on validation accuracy.

## Validation and Model Selection

The original CIFAR-10 training set was divided into:

* **45,000 training images**
* **5,000 validation images**

The model was trained using the 45,000 training images.

The model with the best validation accuracy was selected before performing the final evaluation on the official CIFAR-10 test set.

The best recorded validation accuracy was approximately **75.08%**.

## Analysis

The training accuracy increased progressively during training, while the validation accuracy improved and showed some fluctuation in the later epochs.

The difference between training and validation performance indicates some degree of train-validation separation during later training. Selecting the model based on validation accuracy helps choose the checkpoint with the strongest validation performance rather than simply using the final training epoch.

The final test accuracy of **74.75%** demonstrates that the implemented Vision Transformer was able to learn the CIFAR-10 classification task without using pre-trained weights.

## Reproducibility

The experiment uses **random seed 42**.

The notebook contains the complete:

* Dataset loading and preprocessing
* Train/validation split
* Patch embedding
* Transformer architecture
* Training procedure
* Validation procedure
* Model selection
* Final test evaluation

required to reproduce the experiment in Google Colab.

## Summary

This project demonstrates a complete Vision Transformer implementation from scratch for CIFAR-10 classification using PyTorch, including the core Transformer components and a reproducible training and evaluation pipeline.
