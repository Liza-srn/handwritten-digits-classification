# Deep Learning Project: Alpha Digits and MNIST

## Introduction

This project implements a deep learning pipeline to train and generate images from two datasets:
1. **Binary Alpha Digits** (`binaryalphadigs.mat`), a dataset of binarized characters.
2. **MNIST** (`mnist_all.mat`), a dataset of handwritten digits.

The project explores three types of neural networks:
- **Restricted Boltzmann Machines (RBM):** Probabilistic models for unsupervised feature extraction.
- **Deep Belief Networks (DBN):** Stacks of RBMs for deep representation learning.
- **Deep Neural Networks (DNN):** Fully connected networks for classification tasks, trained using backpropagation.

---

## Mathematical Principles

### 1. **RBM: Restricted Boltzmann Machine**

An RBM is an energy-based probabilistic model consisting of two layers:
- **Visible Layer (\(v\)):** Represents the input data.
- **Hidden Layer (\(h\)):** Captures the latent features.

The energy function of an RBM is defined as:
\[
E(v, h) = -v^T W h - a^T v - b^T h
\]
where:
- \(W\): Weight matrix between visible and hidden layers.
- \(a, b\): Biases of visible and hidden layers.

The goal is to minimize the energy function by learning the parameters \(W, a, b\) using Contrastive Divergence (CD).

Key steps:
- **Forward pass (Sampling \(h\)):**
  \[
  P(h_j = 1 | v) = \sigma(b_j + \sum_i W_{ij} v_i)
  \]
  where \(\sigma(x) = \frac{1}{1 + e^{-x}}\) is the sigmoid function.

- **Backward pass (Reconstructing \(v\)):**
  \[
  P(v_i = 1 | h) = \sigma(a_i + \sum_j W_{ij} h_j)
  \]

- **Parameter update:**
  \[
  W \leftarrow W + \eta (\langle v_0 h_0^T \rangle - \langle v_k h_k^T \rangle)
  \]

### 2. **DBN: Deep Belief Network**

A DBN is a stack of RBMs, trained layer by layer:
1. The first RBM is trained on raw input data to learn features.
2. The second RBM is trained on the activations (hidden outputs) of the first RBM.
3. This process continues for deeper layers.

DBNs leverage greedy layer-wise pretraining to initialize weights, making them effective for deep learning tasks. After pretraining, the DBN can be fine-tuned using supervised learning.

Mathematical steps:
- **Layer-wise training:** Train each RBM using the steps outlined in the RBM section.
- **Fine-tuning:** Perform backpropagation to minimize a supervised loss function:
  \[
  L(\theta) = -\sum_{i} y_i \log(\hat{y}_i)
  \]

### 3. **DNN: Deep Neural Network**

A DNN is a fully connected feedforward network consisting of:
- Input layer (\(x\)).
- One or more hidden layers (\(h\)).
- Output layer (\(\hat{y}\)) for classification.

Key mathematical principles:
- **Forward pass:**
  \[
  h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
  \]
  where \(l\) is the layer index.

- **Output probabilities using softmax:**
  \[
  P(y=k|x) = \frac{\exp(z_k)}{\sum_j \exp(z_j)}
  \]

- **Backpropagation for weight updates:**
  \[
  \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T
  \]
  where \(\delta^{(l)}\) is the error term.

---

## Installation

### Requirements
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy scipy matplotlib pandas scikit-learn

