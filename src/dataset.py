"""
Dataset loaders and utilities for Saree Design Recognition.
Handles:
1. Directory parsing (design-folder structure)
2. Kaggle and DeepLure dataset layouts
3. Synthetic dummy dataset generation for immediate reproducibility & smoke tests
4. Gallery/Query protocol splitting
5. Verification pair generation (positive & negative pairs)
"""

import os
import glob
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np


class SareeDesignDataset(Dataset):
    """
    Standard classification / metric learning dataset.
    Reads images belonging to different saree design classes.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            samples: List of (image_path, class_id)
            transform: torchvision transforms
        """
        self.samples = samples
        self.transform = transform
        self.classes = sorted(list(set(label for _, label in samples)))
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


class VerificationPairDataset(Dataset):
    """
    Dataset returning image pairs for 1:1 Verification evaluation.
    Label is 1 for same design (different colorways), 0 for different design.
    """
    def __init__(self, pairs: List[Tuple[str, str, int]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label


class BalancedBatchSampler(Sampler):
    """
    P-K batch sampler for Deep Metric Learning:
    Selects P unique designs, and K images per design, ensuring each batch has
    rich positive pairs and hard negative contrasts.
    """
    def __init__(self, dataset: SareeDesignDataset, n_classes: int, n_samples: int):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.classes = list(self.dataset.label_to_indices.keys())
        # Filter classes that have at least 2 samples
        self.valid_classes = [c for c in self.classes if len(self.dataset.label_to_indices[c]) >= 2]
        self.n_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            selected_classes = random.sample(self.valid_classes, min(self.n_classes, len(self.valid_classes)))
            for c in selected_classes:
                indices = self.dataset.label_to_indices[c]
                if len(indices) >= self.n_samples:
                    batch.extend(random.sample(indices, self.n_samples))
                else:
                    batch.extend(random.choices(indices, k=self.n_samples))
            yield batch

    def __len__(self):
        return self.n_batches


def scan_dataset_directory(root_dir: str) -> List[Tuple[str, int]]:
    """
    Recursively scans a directory where subdirectories represent design IDs.
    Returns list of (image_path, class_id).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples = []
    
    if not os.path.exists(root_dir):
        return []

    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for class_id, subdir in enumerate(subdirs):
        sub_path = os.path.join(root_dir, subdir)
        for fname in os.listdir(sub_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                samples.append((os.path.join(sub_path, fname), class_id))
                
    return samples


def create_gallery_query_split(samples: List[Tuple[str, int]], 
                               gallery_ratio: float = 0.5, 
                               seed: int = 42) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits evaluation samples into Gallery (reference database) and Query sets.
    Ensures that for each design class in the evaluation set, at least 1 image is in gallery
    and at least 1 image is in query.
    """
    random.seed(seed)
    class_to_samples: Dict[int, List[str]] = {}
    for path, label in samples:
        if label not in class_to_samples:
            class_to_samples[label] = []
        class_to_samples[label].append(path)

    gallery_samples = []
    query_samples = []

    for label, paths in class_to_samples.items():
        if len(paths) < 2:
            # Single sample cannot be evaluated for query-gallery matching
            continue
        random.shuffle(paths)
        n_gallery = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gallery]:
            gallery_samples.append((p, label))
        for p in paths[n_gallery:]:
            query_samples.append((p, label))

    return gallery_samples, query_samples


def generate_verification_pairs(samples: List[Tuple[str, int]], 
                                num_pairs: int = 1000, 
                                seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Generates balanced positive (same design, different image/colorway) 
    and negative (different designs) image pairs for verification evaluation.
    """
    random.seed(seed)
    class_to_paths: Dict[int, List[str]] = {}
    for p, l in samples:
        if l not in class_to_paths:
            class_to_paths[l] = []
        class_to_paths[l].append(p)

    valid_classes = [c for c, paths in class_to_paths.items() if len(paths) >= 2]
    all_classes = list(class_to_paths.keys())

    pairs = []
    half = num_pairs // 2

    # Positive pairs (label = 1)
    for _ in range(half):
        c = random.choice(valid_classes)
        p1, p2 = random.sample(class_to_paths[c], 2)
        pairs.append((p1, p2, 1))

    # Negative pairs (label = 0)
    for _ in range(half):
        c1, c2 = random.sample(all_classes, 2)
        p1 = random.choice(class_to_paths[c1])
        p2 = random.choice(class_to_paths[c2])
        pairs.append((p1, p2, 0))

    random.shuffle(pairs)
    return pairs


def create_synthetic_saree_corpus(output_dir: str, num_designs: int = 15, 
                                  colorways_per_design: int = 4, 
                                  img_size: int = 256):
    """
    Generates a synthetic textile design dataset for instant verification & smoke testing.
    Each design has a distinct geometric/motif structure (e.g. Paisley, Floral Jaal, Temple Borders,
    Chevron, Polka Zari) rendered across totally distinct color palettes (Red-Gold, Blue-Silver,
    Green-Yellow, Black-Gold).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    palettes = [
        # (Background, Primary Motif, Secondary Border)
        ((180, 20, 30), (255, 215, 0), (200, 160, 20)),   # Red & Gold Zari
        ((20, 40, 150), (220, 220, 230), (180, 180, 210)), # Royal Blue & Silver
        ((25, 120, 50), (255, 230, 80), (240, 180, 30)),  # Emerald & Mustard
        ((40, 40, 40), (240, 200, 80), (210, 170, 50)),   # Charcoal & Antique Gold
        ((160, 50, 140), (250, 240, 200), (220, 190, 100)), # Magenta & Cream
        ((220, 110, 30), (40, 40, 40), (255, 220, 100))    # Rust Orange & Copper
    ]

    for d_idx in range(num_designs):
        design_folder = os.path.join(output_dir, f"design_{d_idx:03d}")
        os.makedirs(design_folder, exist_ok=True)

        selected_palettes = random.sample(palettes, min(colorways_per_design, len(palettes)))
        
        for c_idx, (bg_col, motif_col, border_col) in enumerate(selected_palettes):
            img = Image.new("RGB", (img_size, img_size), bg_col)
            draw = ImageDraw.Draw(img)

            # Draw structured geometric motif unique to design d_idx
            # 1. Outer Border
            border_width = 16 + (d_idx % 4) * 4
            draw.rectangle([0, 0, img_size, border_width], fill=border_col)
            draw.rectangle([0, img_size - border_width, img_size, img_size], fill=border_col)
            
            # 2. Distinct Pattern Logic based on design index
            step = 32 + (d_idx % 3) * 16
            motif_type = d_idx % 5

            if motif_type == 0:  # Zari Diamond Buttis
                for x in range(border_width + 10, img_size - border_width - 10, step):
                    for y in range(border_width + 10, img_size - border_width - 10, step):
                        draw.polygon([(x, y-10), (x+10, y), (x, y+10), (x-10, y)], fill=motif_col)

            elif motif_type == 1:  # Temple / Chevron Triangular Motif
                for x in range(0, img_size, 20):
                    draw.polygon([(x, border_width), (x+10, border_width+18), (x+20, border_width)], fill=motif_col)
                    draw.polygon([(x, img_size - border_width), (x+10, img_size - border_width - 18), (x+20, img_size - border_width)], fill=motif_col)
                # Diagonal jaal
                for d in range(0, img_size * 2, step):
                    draw.line([(0, d), (d, 0)], fill=motif_col, width=2)

            elif motif_type == 2:  # Circular Paisley Medallions
                for x in range(border_width + 15, img_size - border_width - 15, step):
                    for y in range(border_width + 15, img_size - border_width - 15, step):
                        draw.ellipse([x-12, y-12, x+12, y+12], outline=motif_col, width=3)
                        draw.ellipse([x-4, y-4, x+4, y+4], fill=motif_col)

            elif motif_type == 3:  # Floral Jaal / Lattice
                for x in range(border_width, img_size - border_width, 24):
                    draw.line([(x, border_width), (x, img_size - border_width)], fill=motif_col, width=2)
                for y in range(border_width, img_size - border_width, 24):
                    draw.line([(border_width, y), (img_size - border_width, y)], fill=motif_col, width=2)

            else:  # Multi-tier horizontal zari stripes & micro dots
                for y in range(border_width + 10, img_size - border_width - 10, step):
                    draw.line([(0, y), (img_size, y)], fill=motif_col, width=3)
                    for x in range(10, img_size, 20):
                        draw.rectangle([x-2, y+10-2, x+2, y+10+2], fill=border_col)

            save_path = os.path.join(design_folder, f"colorway_{c_idx:02d}.jpg")
            img.save(save_path, "JPEG", quality=95)

    print(f"Created synthetic dataset at '{output_dir}': {num_designs} designs, {colorways_per_design} colorways each.")
