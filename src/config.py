import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # Project info
    PROJECT_NAME: str = "DeepLure-Color-Invariant-Saree-Recognition"
    SEED: int = 42
    
    # Dataset configurations
    DATASET_DIR: str = os.environ.get("DATASET_DIR", "./data")
    KAGGLE_DATASET_PATH: str = os.path.join(DATASET_DIR, "indian-saree-patterns")
    DEEPLURE_DATASET_PATH: str = os.path.join(DATASET_DIR, "deeplure_saree_corpus")
    
    # Model configuration
    # Options: 'efficientnet_b0', 'convnext_tiny', 'mobilenetv3_large_100', 'resnet50'
    BACKBONE_NAME: str = "convnext_tiny"
    PRETRAINED: bool = True
    EMBEDDING_DIM: int = 512
    DROP_RATE: float = 0.2
    POOLING_TYPE: str = "gem"  # 'gem' (Generalized Mean) or 'adaptive_avg'
    GEM_P: float = 3.0
    
    # Input image parameters
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    IN_CHANNELS: int = 3
    
    # Color Invariance Strategy
    # Options: 'random_extreme_jitter' (heavy HSV jitter + grayscale mix), 
    # 'structure_guided' (Sobel/Gradient enhancement), 'pure_grayscale'
    COLOR_STRATEGY: str = "random_extreme_jitter"
    GRAYSCALE_PROB: float = 0.4
    HUE_SHIFT_LIMIT: float = 0.5    # Drastic color palette permutation [-180, +180] deg
    SAT_SHIFT_LIMIT: float = 0.5
    VAL_SHIFT_LIMIT: float = 0.3
    
    # Training Hyperparameters
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0  # 0 for Windows, 2 for Linux/Kaggle
    NUM_EPOCHS: int = 15

    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 2
    MIN_LR: float = 1e-6
    USE_AMP: bool = True  # Automatic Mixed Precision for fast Kaggle GPU training
    
    # Metric Learning / ArcFace Loss Parameters
    LOSS_TYPE: str = "arcface"  # 'arcface' or 'triplet'
    ARCFACE_SCALE: float = 30.0   # Feature scale s
    ARCFACE_MARGIN: float = 0.35  # Angular margin m (in radians)
    TRIPLET_MARGIN: float = 0.3
    
    # Evaluation & Protocol
    GALLERY_SPLIT_RATIO: float = 0.5  # Fraction of test images allocated to Gallery
    TOP_K_METRICS: List[int] = field(default_factory=lambda: [1, 5, 10])
    VERIFICATION_NUM_PAIRS: int = 2000  # Number of pos/neg pairs for verification eval
    
    # Outputs
    OUTPUT_DIR: str = "./outputs"
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")

cfg = Config()
