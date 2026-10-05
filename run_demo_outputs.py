"""
Generate complete visual outputs, retrieval demonstrations, and evaluation metrics for DeepLure Saree Recognition.
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg
from src.dataset import (
    scan_dataset_directory, 
    create_synthetic_saree_corpus, 
    create_gallery_query_split, 
    generate_verification_pairs,
    SareeDesignDataset,
    VerificationPairDataset
)
from src.transforms import ColorInvarianceAugment, get_eval_transforms
from src.models import SareeDesignEmbedder, ArcMarginProduct
from src.evaluate import extract_embeddings, evaluate_identification, evaluate_verification
from src.benchmark import run_full_efficiency_benchmark

def main():
    os.makedirs("outputs", exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    synth_dir = os.path.join(cfg.DATASET_DIR, "synthetic_saree_corpus")
    if not os.path.exists(synth_dir) or len(os.listdir(synth_dir)) == 0:
        create_synthetic_saree_corpus(synth_dir, num_designs=25, colorways_per_design=5)

    samples = scan_dataset_directory(synth_dir)
    print(f"[*] Found {len(samples)} total saree images across {len(set(l for _, l in samples))} designs.")

    # 1. VISUAL OUTPUT 1: Multi-Colorway Saree Demonstrator
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    designs = sorted(list(set(l for _, l in samples)))[:3]
    for row_idx, d in enumerate(designs):
        colorway_imgs = [p for p, l in samples if l == d][:5]
        for col_idx, img_path in enumerate(colorway_imgs):
            img = Image.open(img_path)
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f"{d} | Palette #{col_idx+1}", fontsize=11, fontweight='bold')
            axes[row_idx, col_idx].axis('off')
    plt.suptitle("DeepLure Saree Dataset: Same Motif Geometry in Contrasting Colorways", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/1_multicolorway_designs.png", dpi=150)
    plt.close(fig)
    print("[+] Saved outputs/1_multicolorway_designs.png")

    # 2. VISUAL OUTPUT 2: Color-Invariance Augmentation Pipeline
    sample_img_path = samples[0][0]
    orig_img = Image.open(sample_img_path).convert("RGB")
    augmenter = ColorInvarianceAugment(grayscale_prob=0.4)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Original Saree Image", fontsize=11, fontweight='bold', color='navy')
    axes[0, 0].axis('off')

    for i in range(1, 8):
        row = i // 4
        col = i % 4
        aug_img = augmenter(orig_img)
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f"Augmented Variation #{i}", fontsize=11)
        axes[row, col].axis('off')
    plt.suptitle("DeepLure Photometric Decoupling: Extreme Hue Jitter, Solarization & Grayscale", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/2_color_invariance_augmentations.png", dpi=150)
    plt.close(fig)
    print("[+] Saved outputs/2_color_invariance_augmentations.png")

    # 3. Model & Feature Extraction Setup
    device = torch.device("cpu")
    model = SareeDesignEmbedder(
        backbone_name=cfg.BACKBONE_NAME,
        pretrained=False,
        embedding_dim=cfg.EMBEDDING_DIM,
        pooling=cfg.POOLING_TYPE,
        gem_p=cfg.GEM_P
    ).to(device)
    model.eval()

    all_classes = sorted(list(set(l for _, l in samples)))
    n_train = max(1, int(len(all_classes) * 0.8))
    test_classes = set(all_classes[n_train:])
    test_samples = [(p, l) for p, l in samples if l in test_classes]

    gallery_samples, query_samples = create_gallery_query_split(test_samples, gallery_ratio=0.5)
    eval_transform = get_eval_transforms(cfg.IMAGE_SIZE)

    gallery_loader = DataLoader(SareeDesignDataset(gallery_samples, transform=eval_transform), batch_size=8, shuffle=False)
    query_loader = DataLoader(SareeDesignDataset(query_samples, transform=eval_transform), batch_size=8, shuffle=False)

    q_feats, q_labels, q_paths = extract_embeddings(model, query_loader, device)
    g_feats, g_labels, g_paths = extract_embeddings(model, gallery_loader, device)

    # 4. Identification Metrics
    ident_metrics = evaluate_identification(q_feats, q_labels, g_feats, g_labels, topk=[1, 5, 10])

    # 5. Verification Metrics
    verif_pairs = generate_verification_pairs(test_samples, num_pairs=min(len(test_samples) * 4, 300))
    verif_loader = DataLoader(VerificationPairDataset(verif_pairs, transform=eval_transform), batch_size=8, shuffle=False)
    verif_metrics = evaluate_verification(model, verif_loader, device)

    # 6. VISUAL OUTPUT 3: Open-Set Retrieval Showcase (Top-5 Gallery Retrieval)
    sim_matrix = np.dot(q_feats, g_feats.T)
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    num_display = min(3, len(query_samples))

    for q_idx in range(num_display):
        q_path = q_paths[q_idx]
        q_lbl = q_labels[q_idx]
        q_img = Image.open(q_path)

        axes[q_idx, 0].imshow(q_img)
        axes[q_idx, 0].set_title(f"QUERY\nClass: {q_lbl}", fontsize=11, fontweight='bold', color='blue')
        axes[q_idx, 0].axis('off')

        ranked_indices = np.argsort(-sim_matrix[q_idx])[:5]
        for rank, g_idx in enumerate(ranked_indices):
            g_path = g_paths[g_idx]
            g_lbl = g_labels[g_idx]
            sim = sim_matrix[q_idx, g_idx]
            g_img = Image.open(g_path)

            axes[q_idx, rank + 1].imshow(g_img)
            is_match = (q_lbl == g_lbl)
            match_color = "green" if is_match else "red"
            status = "MATCH" if is_match else "MISMATCH"
            axes[q_idx, rank + 1].set_title(
                f"Rank #{rank+1} ({status})\nSim: {sim:.3f}", 
                fontsize=10, 
                fontweight='bold', 
                color=match_color
            )
            for spine in axes[q_idx, rank + 1].spines.values():
                spine.set_edgecolor(match_color)
                spine.set_linewidth(3)
            axes[q_idx, rank + 1].set_xticks([])
            axes[q_idx, rank + 1].set_yticks([])

    plt.suptitle("DeepLure Saree Retrieval Demonstration (Query vs Top-5 Gallery Matches)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/3_retrieval_showcase.png", dpi=150)
    plt.close(fig)
    print("[+] Saved outputs/3_retrieval_showcase.png")

    # 7. Efficiency Benchmarks
    bench_results = run_full_efficiency_benchmark(model, cfg.EMBEDDING_DIM, cfg.IMAGE_SIZE)

    # 8. Print formatted numerical summary to stdout
    print("\n" + "="*70)
    print("                DEEPLURE EXPERIMENTAL OUTPUT REPORT                ")
    print("="*70)
    print("\n[1] OPEN-SET IDENTIFICATION PERFORMANCE (Gallery Ranking)")
    print(f"    • Rank-1 Accuracy:  {ident_metrics['Rank-1']:.2f}%")
    print(f"    • Rank-5 Accuracy:  {ident_metrics['Rank-5']:.2f}%")
    print(f"    • Rank-10 Accuracy: {ident_metrics['Rank-10']:.2f}%")
    print(f"    • Mean Average Precision (mAP): {ident_metrics['mAP']:.2f}%")

    print("\n[2] PAIRWISE VERIFICATION PERFORMANCE (1:1 Authentication)")
    print(f"    • Area Under ROC Curve (ROC-AUC): {verif_metrics['ROC-AUC']:.2f}%")
    print(f"    • Equal Error Rate (EER):         {verif_metrics['EER']:.2f}%")
    print(f"    • Optimal Cosine Threshold:       {verif_metrics['Optimal_Threshold']:.4f}")
    print(f"    • F1-Score at EER:                {verif_metrics['F1-Score']:.2f}%")
    print(f"    • Precision / Recall:             {verif_metrics['Precision']:.2f}% / {verif_metrics['Recall']:.2f}%")

    print("\n[3] EFFICIENCY & RESOURCE BENCHMARKS")
    print(f"    • Total Parameter Count:   {bench_results['total_parameters']:,} ({bench_results['parameters_in_millions']} Million)")
    print(f"    • Computational GFLOPs:    {bench_results['estimated_gflops']:.2f} GFLOPs")
    print(f"    • CPU Inference Latency:   {bench_results['latency']['mean_latency_ms']:.2f} ms/image")
    print(f"    • CPU Throughput:          {bench_results['latency']['throughput_fps']:.1f} FPS")
    print(f"    • Single Embedding Size:   {bench_results['storage']['Single_Vector_FP32']}")
    print(f"    • 10,000 Gallery Storage:  {bench_results['storage']['Gallery_10K_FP32']}")
    print(f"    • 100,000 Gallery Storage: {bench_results['storage']['Gallery_100K_FP32']}")
    print(f"    • 1,000,000 Gallery RAM:   {bench_results['storage']['Gallery_1M_FP32']} (or {bench_results['storage']['Gallery_1M_INT8_Quantized']} with INT8 Quantization)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
