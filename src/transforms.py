import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Tuple, List

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ColorInvarianceAugment:
    """
    Scrambles color palette (Hue shifting, saturation perturbation, solarization, random grayscale)
    while strictly preserving motif contour, border patterns, and woven geometry.
    """
    def __init__(self, grayscale_prob: float = 0.4):
        self.grayscale_prob = grayscale_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Random Channel Shuffle / Palette Inversion (15% chance)
        if random.random() < 0.15:
            r, g, b = img.split()
            channels = [r, g, b]
            random.shuffle(channels)
            img = Image.merge("RGB", channels)

        # 2. Extreme Color / Saturation / Contrast Jitter
        if random.random() < 0.8:
            # Saturation jitter
            sat_factor = random.uniform(0.3, 1.8)
            img = ImageEnhance.Color(img).enhance(sat_factor)
            
            # Contrast jitter
            cont_factor = random.uniform(0.7, 1.4)
            img = ImageEnhance.Contrast(img).enhance(cont_factor)
            
            # Brightness jitter
            bright_factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(bright_factor)

        # 3. Random Grayscale (crucial to enforce pure structural motif learning)
        if random.random() < self.grayscale_prob:
            img = ImageOps.grayscale(img).convert("RGB")

        # 4. Random Solarization (inverts pixels above threshold, decoupling hue)
        if random.random() < 0.2:
            threshold = random.randint(128, 220)
            img = ImageOps.solarize(img, threshold=threshold)

        # 5. Random Contrast Equalization
        if random.random() < 0.25:
            img = ImageOps.autocontrast(img)

        return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Converts PIL Image to normalized PyTorch tensor."""
    arr = np.array(img).astype(np.float32) / 255.0  # H, W, C
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C, H, W
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class CustomCompose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img):
        for t in self.transforms_list:
            img = t(img)
        return img


def get_train_transforms(image_size: Tuple[int, int] = (256, 256), 
                         grayscale_prob: float = 0.4):
    """
    Training pipeline:
    - Geometric resizing & random crop
    - Color Invariance block (Palette permutation, solarization, random grayscale)
    - Normalization
    """
    def resize_and_crop(img: Image.Image) -> Image.Image:
        # Resize slightly larger then random crop
        w, h = image_size
        img_res = img.resize((w + 24, h + 24), Image.Resampling.BILINEAR)
        left = random.randint(0, 24)
        top = random.randint(0, 24)
        img_crop = img_res.crop((left, top, left + w, top + h))
        if random.random() < 0.5:
            img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
        return img_crop

    return CustomCompose([
        resize_and_crop,
        ColorInvarianceAugment(grayscale_prob=grayscale_prob),
        pil_to_tensor
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (256, 256),
                        force_grayscale: bool = False):
    """
    Evaluation pipeline:
    Deterministic resizing, optional grayscale test for zero-shot palette ablation, and standard normalization.
    """
    def eval_resize(img: Image.Image) -> Image.Image:
        img_res = img.resize(image_size, Image.Resampling.BILINEAR)
        if force_grayscale:
            img_res = ImageOps.grayscale(img_res).convert("RGB")
        return img_res

    return CustomCompose([
        eval_resize,
        pil_to_tensor
    ])

