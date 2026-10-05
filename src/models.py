"""
Color-Invariant Saree Design Recognition Models:
- Backbone (timm/torchvision with built-in native PyTorch ConvNeXt-Tiny fallback)
- GeM (Generalized Mean) Pooling for motif feature aggregation
- Compact metric projection head with L2 hypersphere normalization
- ArcFace Margin Product classifier head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class NativeConvNeXtBlock(nn.Module):
    """Native PyTorch 7x7 Depthwise ConvNeXt Block for pure offline execution."""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + x


class NativeLightweightBackbone(nn.Module):
    """Fallback textile feature extractor when external model hubs are offline."""
    def __init__(self, in_chans: int = 3, out_dim: int = 768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 96, kernel_size=4, stride=4),
            nn.GroupNorm(1, 96)
        )
        self.stage1 = nn.Sequential(
            NativeConvNeXtBlock(96),
            NativeConvNeXtBlock(96)
        )
        self.downsample1 = nn.Sequential(
            nn.GroupNorm(1, 96),
            nn.Conv2d(96, 192, kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(
            NativeConvNeXtBlock(192),
            NativeConvNeXtBlock(192)
        )
        self.downsample2 = nn.Sequential(
            nn.GroupNorm(1, 192),
            nn.Conv2d(192, out_dim, kernel_size=2, stride=2)
        )
        self.stage3 = nn.Sequential(
            NativeConvNeXtBlock(out_dim),
            NativeConvNeXtBlock(out_dim)
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        return x


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling:
    f(x) = (1/|X| * sum(x^p))^(1/p)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else torch.tensor(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamp = x.clamp(min=self.eps)
        p = self.p.to(x.device)
        return F.adaptive_avg_pool2d(x_clamp.pow(p), (1, 1)).pow(1.0 / p)


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss Head
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35, 
                 easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SareeDesignEmbedder(nn.Module):
    """
    End-to-end model for extracting color-invariant textile motif embeddings.
    """
    def __init__(self, 
                 backbone_name: str = "convnext_tiny", 
                 pretrained: bool = True, 
                 embedding_dim: int = 512, 
                 drop_rate: float = 0.2,
                 pooling: str = "gem",
                 gem_p: float = 3.0):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        
        if HAS_TIMM:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                features_only=True
            )
            dummy_in = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                feat_maps = self.backbone(dummy_in)
                last_feat = feat_maps[-1]
                in_features = last_feat.shape[1]
            self.use_timm = True
        else:
            # Resilient native PyTorch fallback
            in_features = 768
            self.backbone = NativeLightweightBackbone(in_chans=3, out_dim=in_features)
            self.use_timm = False

        if pooling == "gem":
            self.pooling = GeMPooling(p=gem_p, learnable=True)
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_timm:
            feat_maps = self.backbone(x)
            feat = feat_maps[-1]
        else:
            feat = self.backbone(x)

        pooled = self.pooling(feat)
        flat = torch.flatten(pooled, 1)
        emb = self.neck(flat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

