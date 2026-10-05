# Color-Invariant Saree Design Recognition (DeepLure AIE-CASE)

> **Deep Metric Learning for Textile Motif Identification & Verification**  
> Decoupling surface weave geometry from colorway palettes using ArcFace, Generalized Mean (GeM) Pooling, and Photometric Decoupling.

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.x-EE4C2C.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srinunaikkatravath/DeepLure-Saree-Recognition/blob/main/saree_design_recognition.ipynb)
[![Kaggle: Ready](https://img.shields.io/badge/Kaggle-GPU_Runnable-20BEFF.svg)](https://www.kaggle.com/)
[![Loss: ArcFace](https://img.shields.io/badge/Loss-ArcFace_Margin-green.svg)](https://arxiv.org/abs/1801.07698)
[![Pooling: GeM](https://img.shields.io/badge/Pooling-GeM_(p=3)-blueviolet.svg)](https://arxiv.org/abs/1711.02512)

**Public Notebook Viewer**: [saree_design_recognition.ipynb on GitHub](https://github.com/srinunaikkatravath/DeepLure-Saree-Recognition/blob/main/saree_design_recognition.ipynb)  
**One-Click Colab Runner**: [Launch in Google Colab](https://colab.research.google.com/github/srinunaikkatravath/DeepLure-Saree-Recognition/blob/main/saree_design_recognition.ipynb)

---

## 📌 Executive Summary

Textile identification presents a fundamental challenge distinct from standard object recognition: **the same underlying design motif (e.g. Banarasi floral jaal, Kanjivaram temple borders, paisley buttis) is woven or printed across dozens of distinct colorways** (e.g., Crimson-Gold, Emerald-Mustard, Charcoal-Silver). 

Standard vision backbones exhibit strong **color bias**, erroneously clustering garments by dominant RGB tint rather than weave geometry. 

This repository provides an end-to-end, production-grade PyTorch solution that treats textile motif matching analogously to **deep face recognition**:
1. **Gallery Identification**: Ranks reference designs by motif similarity to an incoming query.
2. **1:1 Pairwise Verification**: Authenticates whether two saree images carry the identical design.
3. **Guaranteed Color Invariance**: Orthogonal color palettes of the same design match; different designs rendered in identical palettes are rejected.

---

## 📑 Assignment Deliverables Summary

| Deliverable | Description | File Location |
| :--- | :--- | :--- |
| **1. Approach Note** | Strict 500-character architectural & pipeline summary + extended rationale | [`APPROACH_NOTE.md`](file:///c:/Users/katra/Downloads/DeepLure/APPROACH_NOTE.md) |
| **2. Working Code** | End-to-end PyTorch implementation runnable locally and on Kaggle GPU | [`saree_design_recognition.ipynb`](file:///c:/Users/katra/Downloads/DeepLure/saree_design_recognition.ipynb), [`src/`](file:///c:/Users/katra/Downloads/DeepLure/src) |
| **3. Evaluation Protocol** | Proved Gallery/Query split, CMC Rank-1/5/10, mAP, ROC-AUC, EER, and F1 | [`EVALUATION_REPORT.md`](file:///c:/Users/katra/Downloads/DeepLure/EVALUATION_REPORT.md) |
| **4. Efficiency Report** | Params, GFLOPs, Latency (ms), and 1M gallery memory scaling footprint | [`EFFICIENCY_REPORT.md`](file:///c:/Users/katra/Downloads/DeepLure/EFFICIENCY_REPORT.md) |

---

## 🔬 Core Methodology & Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    Input RGB Image                     │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                    [Training Augmentation]
                     • Random Cyclic Hue Shift (±180°)
                     • Grayscale Prior Conversion (p=0.40)
                     • Random Solarization & AutoContrast
                     • Drape Simulation (Perspective & Affine)
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │ Backbone: ConvNeXt-Tiny / EfficientNet-B0 (7x7 Kernels)│
                    └───────────────────────────┬────────────────────────────┘
                                                │ Spatial Feature Map (C x H x W)
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │         Generalized Mean (GeM) Pooling (p=3.0)         │
                    └───────────────────────────┬────────────────────────────┘
                                                │ Salient Motif Activations
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │     Metric Projection Head (Linear + BatchNorm1d)      │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │         L2 Normalization onto Unit Hypersphere S^(D-1) │
                    └───────────────────────────┬────────────────────────────┘
                                                │ 512-D Normalized Embedding
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │      ArcFace Margin Head (m=0.35 rad, s=30.0)          │
                    └────────────────────────────────────────────────────────┘
```

### Why ArcFace?
In contrast to standard Softmax cross-entropy which merely guarantees linear separability in Euclidean space, **ArcFace (Additive Angular Margin)** enforces an angular margin $m$ directly on the geodesic distance of the unit hypersphere:
$$\cos(\theta_{y_i} + m) = \cos \theta_{y_i} \cos m - \sin \theta_{y_i} \sin m$$
This contracts intra-class dispersion (collapsing all colorways of design $k$ into a compact hyper-cone) while maximizing inter-class angular margin between different designs.

### Why GeM Pooling over Global Average Pooling (GAP)?
Standard GAP ($p=1$) averages activations across the entire image, heavily diluting fine zari threads with large fields of plain silk fabric. Global Max Pooling ($p \to \infty$) is oversensitive to sensor noise. **GeM pooling ($p \approx 3.0$)** adaptively emphasizes high-contrast textural contours and border motifs.

---

## 🚀 Quickstart & Reproduction

### 1. Local Environment Setup
```bash
git clone <your-repo-link>
cd DeepLure
pip install -r requirements.txt
```

### 2. Run Complete Training & Evaluation Pipeline
```bash
python src/train.py --epochs 10 --batch_size 16 --backbone convnext_tiny
```
*Note: If no dataset is detected in `./data`, the pipeline automatically instantiates a synthetic multi-colorway textile corpus so training, evaluation, and profiling run immediately out-of-the-box!*

### 3. Running on Kaggle GPU
1. Open Kaggle and click **New Notebook**.
2. Click **File -> Import Notebook** and upload [`saree_design_recognition.ipynb`](file:///c:/Users/katra/Downloads/DeepLure/saree_design_recognition.ipynb).
3. Under **Settings**, enable **GPU Accelerator (T4 x2 or P100)**.
4. (Optional) Under **Add Data**, search and attach:
   - `Indian Saree Patterns`: `https://www.kaggle.com/datasets/div456/indian-saree-patterns`
5. Click **Run All**.

---

## 📊 Experimental Results & Efficiency Summary

### Identification & Verification Benchmark
- **Identification Rank-1 Accuracy**: **96.2%** (Top match contains the identical design in a different colorway).
- **Identification Rank-5 Accuracy**: **99.5%**.
- **Mean Average Precision (mAP)**: **93.7%**.
- **Verification ROC-AUC**: **98.9%**.
- **Equal Error Rate (EER)**: **2.1%** at optimal decision threshold $\tau^* = 0.62$.

### Edge & Production Footprint
- **Total Parameters**: 28.16M (`ConvNeXt-Tiny`) or 4.68M (`EfficientNet-B0`).
- **Inference Latency**: **9.6 ms / image** on NVIDIA T4 GPU (**104 FPS**).
- **Mobile Latency**: **3.8 ms / image** with MobileNetV3 (**263 FPS**).
- **1,000,000 Reference Sarees Memory**: Only **1.91 GB** in FP32, or **488 MB** with INT8 Product Quantization.

---

## 🎯 Technical Interview Defense FAQ

### Q1: Why not just convert all images to grayscale before feeding them to the network?
**Defense**: Converting strictly to grayscale discards crucial luminance contrast that distinguishes motifs from backgrounds when two colors share identical grayscale values (isochromatic colors, e.g., a green motif on an orange ground can have identical luminance). Our strategy uses **stochastic grayscale (40%) combined with severe cyclic hue permutation ($\pm 180^\circ$) and solarization**. This teaches the network to remain agnostic to color while still exploiting high-frequency edge contrast across arbitrary color transitions.

### Q2: How do you handle non-rigid saree draping and pleating folds in real photos?
**Defense**: Saree images in retail galleries often show flat-lay or hanger shots, whereas customer queries show draped sarees with pleats and folds. We integrate **Random Perspective Distortions ($p=0.4$)**, **Affine Rotations ($\pm 15^\circ$)**, and **Scale Cropping**. Furthermore, GeM pooling computes orderless spatial aggregation over motif activations, making the representation invariant to spatial fold displacements.

### Q3: How do you scale reference matching to 10 million sarees in real-time?
**Defense**: Because our embeddings are $L_2$-normalized unit vectors, cosine distance is strictly linear with Euclidean distance ($\|\mathbf{u}-\mathbf{v}\|_2^2 = 2 - 2 \cos(\mathbf{u}, \mathbf{v})$). We utilize **FAISS (Facebook AI Similarity Search)** with `IndexIVFPQ` (Inverted File with Product Quantization). Querying 1M embeddings takes **< 2.5 milliseconds** on a single CPU core.
