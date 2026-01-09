# MSNCVGA: Multi-Scale Network with Cross-View Gated Attention for HAR

This repository contains the official implementation of the paper: **"A Multi-Scale Network with Cross-View Gated Attention Enhancement for Human Activity Recognition"**.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)

## 📄 Paper Information

* **Title:** A Multi-Scale Network with Cross-View Gated Attention Enhancement for Human Activity Recognition
* **Authors:** Hao Wang, Fangyu Liu, Xiang Li, Yiqian Zheng, Ye Li, Fangmin Sun
* **Journal:** IEEE Transactions on Consumer Electronics (TCE)
* **Created Time:** January 2026

## 📖 Abstract

Human Activity Recognition (HAR) based on inertial sensors plays a pivotal role in consumer electronics. However, achieving robust HAR performance remains challenging due to unconstrained environmental noise and large variations in activity duration. 

In this work, we propose **MSNCVGA**, a multi-scale network with cross-view gated attention. The proposed approach explicitly models sequence scale diversity through **Multi-Scale Fusion**, while a **Cross-View Gated Attention (CVGA)** mechanism jointly captures complementary dependencies across **Channel**, **Sequence**, and **Sample** dimensions without excessive information loss.

**Key Contributions:**
1.  **Multi-Scale Fusion Branch:** Explicitly models interactions across temporal scales to reduce redundancy.
2.  **Multi-View Attention Unit (MVAU):** Captures global dependencies from Channel, Sequence, and Sample views.
3.  **Gated Spatial Attention Unit (GSAU):** Refines local features and suppresses noise without aggressive dimensionality reduction.

## 🏗️ Model Architecture

The MSNCVGA model consists of four core components:
1.  **Multi-Scale Block (MSB):** Parallel branches with kernel sizes of 3, 5, and 7 to capture multi-granularity patterns.
2.  **Multi-Scale Fusion Branch:** Adaptive integration of cross-scale information.
3.  **Cross-View Gated Attention (CVGA):** Composed of MVAU (Global) and GSAU (Local Refinement).
4.  **Bi-GRU & Classification Head:** Captures long-range temporal dependencies.

> *Note: With only **0.54M parameters**, the model is highly suitable for resource-constrained edge devices.*

## 📊 Experiments & Results

We evaluated our model on five public datasets using a 5-fold cross-validation strategy.

| Dataset | Accuracy (ACC) | F1-Score |
| :--- | :---: | :---: |
| **PAMAP2** | 0.845 | **0.848** |
| **UCI HAR** | 0.989 | **0.989** |
| **mHealth** | 0.985 | **0.985** |
| **RealWorld** | 0.611 | **0.642** |
| **KU-HAR** | 0.979 | **0.979** |

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Scikit-learn
* Numpy
* Pandas

### Installation
```bash
git clone [https://github.com/your-username/MSNCVGA.git](https://github.com/your-username/MSNCVGA.git)
cd MSNCVGA
pip install -r requirements.txt
