"""
Evaluation Protocol for Saree Design Recognition:
1. Identification:
   - Evaluates Query against Gallery reference database using Cosine Similarity.
   - Cumulative Matching Characteristics (CMC): Rank-1, Rank-5, Rank-10.
   - Mean Average Precision (mAP).
2. Verification:
   - 1:1 Pairwise matching.
   - ROC-AUC (Receiver Operating Characteristic Area Under Curve).
   - Equal Error Rate (EER) and Optimal Decision Threshold.
   - Precision, Recall, and F1-score.
3. Color Invariance Diagnostic (Stress Test):
   - Compares cosine similarity gap between same-design-cross-colorway vs different-design-same-colorway.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, 
                       dataloader: torch.utils.data.DataLoader, 
                       device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts L2-normalized embeddings for all images in the dataloader.
    Returns:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        paths: list of image paths
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []

    for images, labels, paths in dataloader:
        images = images.to(device)
        feats = model(images)
        all_embeddings.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels, all_paths


def evaluate_identification(query_feats: np.ndarray, 
                            query_labels: np.ndarray, 
                            gallery_feats: np.ndarray, 
                            gallery_labels: np.ndarray, 
                            topk: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Evaluates Gallery Retrieval / Identification performance.
    """
    # Dot product of L2-normalized vectors = Cosine similarity
    sim_matrix = np.dot(query_feats, gallery_feats.T)  # (N_query, N_gallery)
    
    # Sort gallery indices by descending similarity
    ranked_indices = np.argsort(-sim_matrix, axis=1)

    cmc_hits = {k: 0 for k in topk}
    aps = []

    num_queries = len(query_labels)

    for i in range(num_queries):
        q_label = query_labels[i]
        retrieved_labels = gallery_labels[ranked_indices[i]]

        # Matches vector (1 where gallery label matches query label)
        matches = (retrieved_labels == q_label).astype(np.float32)

        # CMC Rank-k
        for k in topk:
            if np.sum(matches[:k]) > 0:
                cmc_hits[k] += 1

        # Average Precision (AP)
        num_pos = np.sum(matches)
        if num_pos == 0:
            continue
        
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precision_at_k = cum_matches / ranks
        ap = np.sum(precision_at_k * matches) / num_pos
        aps.append(ap)

    results = {}
    for k in topk:
        results[f"Rank-{k}"] = (cmc_hits[k] / num_queries) * 100.0
    results["mAP"] = (np.mean(aps) if len(aps) > 0 else 0.0) * 100.0

    return results


def evaluate_verification(model: torch.nn.Module, 
                          pair_dataloader: torch.utils.data.DataLoader, 
                          device: torch.device) -> Dict[str, float]:
    """
    Evaluates 1:1 pairwise design verification (same design vs different design).
    """
    model.eval()
    similarities = []
    ground_truths = []

    with torch.no_grad():
        for img1, img2, labels in pair_dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)

            feat1 = model(img1)
            feat2 = model(img2)

            sim = F.cosine_similarity(feat1, feat2).cpu().numpy()
            similarities.extend(sim.tolist())
            ground_truths.extend(labels.numpy().tolist())

    y_score = np.array(similarities)
    y_true = np.array(ground_truths)

    # Compute ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr) * 100.0

    # Compute Equal Error Rate (EER) where FPR == 1 - TPR (or FNR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_idx] * 100.0
    optimal_threshold = thresholds[eer_idx]

    # Metrics at optimal EER threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "ROC-AUC": roc_auc,
        "EER": eer,
        "Optimal_Threshold": float(optimal_threshold),
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1-Score": f1 * 100.0
    }
