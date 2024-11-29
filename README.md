# Deep Learning Project: Alpha Digits and MNIST

## Introduction
This project implements a deep learning pipeline to train and generate images using two datasets:
1. **Binary Alpha Digits** (`binaryalphadigs.mat`): A dataset of binarized characters 
2. **MNIST** (`mnist_all.mat`): A dataset of handwritten digits

The project explores three neural network architectures:
- **Restricted Boltzmann Machines (RBM)**: Probabilistic models for unsupervised feature extraction
- **Deep Belief Networks (DBN)**: Stacks of RBMs for deep representation learning  
- **Deep Neural Networks (DNN)**: Fully connected networks for classification tasks

## Mathematical Principles

### 1. RBM: Restricted Boltzmann Machine
An RBM is an energy-based probabilistic model with two layers:
- **Visible Layer ($v$)**: Represents the input data
- **Hidden Layer ($h$)**: Captures the latent features

The energy function of an RBM is defined as:

$E(v, h) = -v^T W h - a^T v - b^T h$

where:
- $W$: Weight matrix between visible and hidden layers
- $a, b$: Biases of visible and hidden layers

Key steps:
- Forward pass (Sampling $h$):
$P(h_j = 1 | v) = \sigma(b_j + \sum_i W_{ij} v_i)$

- Backward pass (Reconstructing $v$):
$P(v_i = 1 | h) = \sigma(a_i + \sum_j W_{ij} h_j)$

- Parameter update:
$W \leftarrow W + \eta (\langle v_0 h_0^T \rangle - \langle v_k h_k^T \rangle)$

### 2. DBN: Deep Belief Network 
A DBN stacks multiple RBMs, trained layer by layer:
1. First RBM trains on raw input data
2. Subsequent RBMs train on previous layer's activations
3. Process continues for deeper layers

Mathematical steps:
- Layer-wise training: Apply RBM training for each layer
- Fine-tuning: Backpropagation with supervised loss function:
$L(\theta) = -\sum_{i} y_i \log(\hat{y}_i)$

### 3. DNN: Deep Neural Network
A fully connected feedforward network with:
- Input layer ($x$)
- Hidden layers ($h$)
- Output layer ($\hat{y}$)

Key equations:
- Forward pass:
$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$

- Output probabilities (softmax):
$P(y=k|x) = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$

- Backpropagation:
$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$

## Installation

### Requirements
- Python 3.8+
- Required libraries:
```bash
pip install numpy scipy matplotlib pandas scikit-learn

