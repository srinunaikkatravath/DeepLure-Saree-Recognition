# APPROACH NOTE: Color-Invariant Saree Design Recognition

## 1. Official 500-Character Submission Note
*(Formatted to be strictly under 500 characters for concise executive review / form submission)*

```text
To decouple motif geometry from chroma, we adopt a ConvNeXt-Tiny/EfficientNet backbone with learnable Generalized Mean (GeM, p=3) pooling and 512-D L2-normalized hyperspherical projection. Training employs an aggressive color-invariance pipeline: severe HSV hue permutation (±180°), random grayscale (40%), and solarization to suppress palette bias, paired with Sub-center ArcFace loss (s=30, m=0.35) and P-K balanced sampling to enforce compact intra-motif and wide inter-motif angular separation.
```
*(Exact Character Count: 497 characters including spaces)*

---

## 2. Extended Architectural & Methodological Justification
*(Comprehensive technical rationale for the live follow-up interview)*

### 2.1 The Fundamental Challenge: Texture vs. Palette Disentanglement
Standard deep convolutional neural networks and Vision Transformers pre-trained on ImageNet exhibit strong "color bias": representations are heavily dominated by high-energy RGB distribution rather than fine structural frequency. In sarees, the identical weave (e.g., Banarasi floral jaal or Kanjivaram temple borders) is produced in dozens of contrasting colorways (e.g., Red-Gold vs. Emerald-Mustard). A naive cosine similarity will match sarees of the same color palette over sarees of the same design.

### 2.2 Preprocessing & Color-Invariance Pipeline
1. **Hue-Space Permutation**: Random cyclic shift of hue channel in HSV space by up to $\pm 180^\circ$. By randomizing palette relationships, the network cannot rely on color co-occurrence to minimize loss.
2. **Structural Prior via Stochastic Grayscale**: 40% probability of converting training inputs to luminance-only (grayscale). This forces convolutional filters to activate on edges, gradients, and micro-textures.
3. **Photometric Destruction**: Solarization and AutoContrast invert or flatten color gradients while preserving motif contours and boundary transitions.
4. **Geometric Drape Simulation**: Random perspective distortions, affine transformations, and rotations simulate realistic saree draping, folds, and variable camera angles.

### 2.3 Backbone & Feature Aggregation
- **Backbone**: `ConvNeXt-Tiny` (28.6M params) or `EfficientNet-B0` (5.3M params).
  - *ConvNeXt's 7x7 depthwise convolutions* provide a wide receptive field at high spatial resolutions, preserving subtle thread weaves and zari motif outlines.
  - *EfficientNet-B0* serves as the ultra-lean edge deployment option (0.39 GFLOPs).
- **GeM Pooling (Generalized Mean)**:
  $$f_{\text{GeM}}(x) = \left( \frac{1}{|\mathcal{X}|} \sum_{u \in \mathcal{X}} x_u^p \right)^{\frac{1}{p}}$$
  Standard Global Average Pooling ($p=1$) dilutes localized motif signals across large background fabric areas. Global Max Pooling ($p \to \infty$) is hypersensitive to noise and sensor artifacts. Learnable GeM ($p \approx 3.0$) concentrates activations on salient border motifs and recurring buttis.

### 2.4 Metric Learning Objective: ArcFace (Additive Angular Margin Loss)
Inspired by deep face recognition (which DeepLure explicitly references):
$$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
- **Hyperspherical Constraint**: Both embedding vectors $\mathbf{x} \in \mathbb{R}^D$ and class weight prototypes $\mathbf{w}_j \in \mathbb{R}^D$ are $L_2$-normalized onto the unit hypersphere $\mathbb{S}^{D-1}$.
- **Additive Angular Margin ($m = 0.35$ rad, $s = 30$)**: Forces intra-design angular dispersion to contract while penalizing small inter-design angular differences, creating a geometrically partitioned metric space where colorways of design $A$ cluster together tightly, cleanly separated from design $B$.
