"""
Loss functions for Deep Metric Learning:
1. ArcFace Loss (Angular Margin Cross Entropy)
2. Supervised Contrastive Loss (SupCon)
3. Triplet Margin Loss with Cosine Distance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning (Khosla et al., NeurIPS 2020).
    Pulls representations of the same saree design across different colorways together,
    while pushing apart representations from different saree designs.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized embeddings of shape (batch_size, embed_dim)
            labels: Ground truth class IDs of shape (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # Loss
        loss = -mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        return loss


class TripletCosineLoss(nn.Module):
    """
    Triplet loss using cosine distance on L2-normalized embeddings:
    dist(a, b) = 1 - cosine_sim(a, b)
    L = max(0, dist(a, p) - dist(a, n) + margin)
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = 1.0 - F.cosine_similarity(anchor, positive)
        neg_dist = 1.0 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
