# Deep Learning Project: Alpha Digits and MNIST

## Introduction

This project implements a deep learning pipeline for training and generating images based on binary character data (`binaryalphadigs.mat`) and handwritten digits (`MNIST`). The project explores the use of **Restricted Boltzmann Machines (RBM)**, **Deep Belief Networks (DBN)**, and **Deep Neural Networks (DNN)** for representation learning, image generation, and classification.

---

## Features

### 1. **RBM: Restricted Boltzmann Machine**
- Learns binary character representations.
- Supports:
  - Supervised training.
  - Image generation.
  - Hyperparameter variation experiments:
    - Number of characters.
    - Number of neurons in the hidden layer.
    - Number of Gibbs sampling iterations.

### 2. **DBN: Deep Belief Network**
- Stacks multiple RBMs for deeper representation learning.
- Capabilities:
  - Unsupervised pretraining.
  - Image generation.
  - Configurable number of layers and neurons.

### 3. **DNN: Deep Neural Network**
- Fully connected feedforward neural network.
- Experiments:
  - Pretrained (DBN-based) vs. non-pretrained models.
  - Varying data size, number of layers, and neurons.
  - Classification of MNIST data.

---

## Installation

### Requirements
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy scipy matplotlib pandas scikit-learn
