import os
import sys
import argparse
import random

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import cfg
from src.transforms import get_train_transforms, get_eval_transforms

from src.dataset import (
    SareeDesignDataset,
    VerificationPairDataset,
    scan_dataset_directory,
    create_gallery_query_split,
    generate_verification_pairs,
    create_synthetic_saree_corpus
)
from src.models import SareeDesignEmbedder, ArcMarginProduct
from src.evaluate import extract_embeddings, evaluate_identification, evaluate_verification
from src.benchmark import run_full_efficiency_benchmark


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, 
                    arcface_head: nn.Module, 
                    dataloader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    scaler: torch.cuda.amp.GradScaler, 
                    device: torch.device, 
                    use_amp: bool = True) -> float:
    model.train()
    arcface_head.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="DeepLure Color-Invariant Saree Recognition")
    parser.add_argument("--data_dir", type=str, default=cfg.DATASET_DIR, help="Path to saree dataset directory")
    parser.add_argument("--backbone", type=str, default=cfg.BACKBONE_NAME, help="Model backbone name")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=cfg.OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 1. Dataset Discovery & Loading
    # Check if user data exists; if not, generate synthetic saree corpus for full runnable flow
    samples = scan_dataset_directory(args.data_dir)
    if len(samples) == 0:
        synth_dir = os.path.join(args.data_dir, "synthetic_saree_corpus")
        if not os.path.exists(synth_dir) or len(os.listdir(synth_dir)) == 0:
            print("[!] No dataset found in data_dir. Generating synthetic multi-colorway saree corpus...")
            create_synthetic_saree_corpus(synth_dir, num_designs=25, colorways_per_design=5)
        samples = scan_dataset_directory(synth_dir)

    print(f"[*] Loaded {len(samples)} total images across {len(set(l for _, l in samples))} design classes.")

    # Split into Train (80%) and Test (20%) design sets
    all_classes = sorted(list(set(l for _, l in samples)))
    random.shuffle(all_classes)
    n_train = max(1, int(len(all_classes) * 0.8))
    train_classes = set(all_classes[:n_train])
    test_classes = set(all_classes[n_train:])

    # Re-index train classes 0..N_train-1 for ArcFace
    train_class_map = {orig: new for new, orig in enumerate(train_classes)}
    train_samples = [(p, train_class_map[l]) for p, l in samples if l in train_classes]
    test_samples = [(p, l) for p, l in samples if l in test_classes]

    print(f"[*] Train set: {len(train_samples)} images ({len(train_classes)} classes)")
    print(f"[*] Test set: {len(test_samples)} images ({len(test_classes)} classes)")

    # 2. Prepare DataLoaders
    train_transform = get_train_transforms(cfg.IMAGE_SIZE, grayscale_prob=cfg.GRAYSCALE_PROB)
    eval_transform = get_eval_transforms(cfg.IMAGE_SIZE)

    train_dataset = SareeDesignDataset(train_samples, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(args.batch_size, len(train_dataset)), 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        drop_last=True if len(train_dataset) > args.batch_size else False
    )

    # Prepare Gallery & Query sets for Identification Protocol
    gallery_samples, query_samples = create_gallery_query_split(test_samples, gallery_ratio=0.5)
    gallery_loader = DataLoader(
        SareeDesignDataset(gallery_samples, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False
    )
    query_loader = DataLoader(
        SareeDesignDataset(query_samples, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Prepare Verification Pairs (1:1 Verification Protocol)
    verif_pairs = generate_verification_pairs(test_samples, num_pairs=min(len(test_samples) * 4, 1000))
    verif_loader = DataLoader(
        VerificationPairDataset(verif_pairs, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 3. Model & Loss Setup
    model = SareeDesignEmbedder(
        backbone_name=args.backbone,
        pretrained=cfg.PRETRAINED,
        embedding_dim=cfg.EMBEDDING_DIM,
        drop_rate=cfg.DROP_RATE,
        pooling=cfg.POOLING_TYPE,
        gem_p=cfg.GEM_P
    ).to(device)

    num_train_classes = len(train_classes)
    arcface_head = ArcMarginProduct(
        in_features=cfg.EMBEDDING_DIM,
        out_features=num_train_classes,
        s=cfg.ARCFACE_SCALE,
        m=cfg.ARCFACE_MARGIN
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.MIN_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")

    # 4. Training Loop
    best_rank1 = 0.0
    print("\n--- Starting Model Training with Color-Invariant Objectives ---")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, arcface_head, train_loader, criterion, optimizer, scaler, device, use_amp=cfg.USE_AMP
        )
        scheduler.step()

        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Training Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Evaluate every 5 epochs or final epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            if len(query_samples) > 0 and len(gallery_samples) > 0:
                q_feats, q_labels, _ = extract_embeddings(model, query_loader, device)
                g_feats, g_labels, _ = extract_embeddings(model, gallery_loader, device)
                ident_metrics = evaluate_identification(q_feats, q_labels, g_feats, g_labels, topk=[1, 5])
                print(f"    -> [Identification] Rank-1: {ident_metrics['Rank-1']:.2f}% | Rank-5: {ident_metrics['Rank-5']:.2f}% | mAP: {ident_metrics['mAP']:.2f}%")

                if ident_metrics['Rank-1'] > best_rank1:
                    best_rank1 = ident_metrics['Rank-1']
                    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_saree_model.pth"))

            if len(verif_pairs) > 0:
                verif_metrics = evaluate_verification(model, verif_loader, device)
                print(f"    -> [Verification]   ROC-AUC: {verif_metrics['ROC-AUC']:.2f}% | EER: {verif_metrics['EER']:.2f}% | F1: {verif_metrics['F1-Score']:.2f}%")

    print("\n--- Running Final Evaluation & Benchmarks ---")
    # Save final model
    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "final_saree_model.pth"))

    # Efficiency Report
    bench_results = run_full_efficiency_benchmark(model, cfg.EMBEDDING_DIM, cfg.IMAGE_SIZE)
    print("\nEfficiency Report (Deliverable 4):")
    print(f"  • Total Parameters: {bench_results['total_parameters']:,} ({bench_results['parameters_in_millions']} M)")
    print(f"  • Estimated GFLOPs: {bench_results['estimated_gflops']} GFLOPs")
    print(f"  • Mean Latency:     {bench_results['latency']['mean_latency_ms']} ms/image ({bench_results['latency']['throughput_fps']} FPS on {device.type})")
    print(f"  • Embedding Size:   {bench_results['storage']['Single_Vector_FP32']} (512 float32)")
    print(f"  • 1M Gallery RAM:   {bench_results['storage']['Gallery_1M_FP32']}")


if __name__ == "__main__":
    main()
