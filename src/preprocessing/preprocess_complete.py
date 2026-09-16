"""
=============================================================================
PREPROCESSING MODULE - Tomato Disease Detection
=============================================================================

This module handles:
1. Dataset structure validation
2. YOLO format annotation reading
3. Image loading and transformation
4. PyTorch Dataset class for training
5. DataLoader creation

Dataset Format (YOLO):
    images/train/*.jpg
    labels/train/*.txt (format: class_id x_center y_center width height)
    images/val/*.jpg
    labels/val/*.txt
    
Classes (10):
    0: Bacterial spot
    1: Early blight
    2: Late blight
    3: Leaf Mold
    4: Septoria leaf spot
    5: Spider mites
    6: Target Spot
    7: Tomato Yellow Leaf Curl Virus
    8: Tomato mosaic virus
    9: Healthy
"""

from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "tomato_yolo_dataset"

IMAGE_TRAIN = DATASET_PATH / "images" / "train"
IMAGE_VAL = DATASET_PATH / "images" / "val"
LABEL_TRAIN = DATASET_PATH / "labels" / "train"
LABEL_VAL = DATASET_PATH / "labels" / "val"

# Image settings
IMAGE_SIZE = 480  # Input size for YOLOv8 (divisible by 32)
NUM_CLASSES = 10

# Class information
CLASS_NAMES = {
    0: "Bacterial spot",
    1: "Early blight",
    2: "Late blight",
    3: "Leaf Mold",
    4: "Septoria leaf spot",
    5: "Spider mites",
    6: "Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato mosaic virus",
    9: "Healthy",
}

# ImageNet normalization (standard for pretrained models)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================================
# STEP 1: VALIDATE DATASET STRUCTURE
# ============================================================================

def check_dataset_structure() -> bool:
    """
    STEP 1: Verify YOLO dataset structure exists
    
    Checks:
        - dataset/tomato_yolo_dataset/
        - images/train and images/val
        - labels/train and labels/val
    
    Returns:
        bool: True if structure is valid
    
    Raises:
        FileNotFoundError: If any required directory missing
    """
    print("\n" + "="*70)
    print("STEP 1: Validating Dataset Structure")
    print("="*70)
    
    paths = {
        "Dataset root": DATASET_PATH,
        "Train images": IMAGE_TRAIN,
        "Val images": IMAGE_VAL,
        "Train labels": LABEL_TRAIN,
        "Val labels": LABEL_VAL,
    }
    
    all_exist = True
    for name, path in paths.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        
        if not exists:
            all_exist = False
    
    if not all_exist:
        raise FileNotFoundError(
            f"\n❌ Dataset structure incomplete!\n"
            f"Expected dataset at: {DATASET_PATH}"
        )
    
    print("✓ Dataset structure validated!\n")
    return True


# ============================================================================
# STEP 2: COLLECT IMAGE FILES
# ============================================================================

def get_image_files(image_directory: Path) -> List[Path]:
    """
    STEP 2: Get all supported image files from directory
    
    Supports: .jpg, .jpeg, .png, .bmp, .webp
    
    Args:
        image_directory: Path to image folder
        
    Returns:
        List of sorted image paths
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    image_files = [
        path for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    
    return sorted(image_files)


# ============================================================================
# STEP 3: LOAD YOLO ANNOTATIONS
# ============================================================================

def read_yolo_label(label_path: Path) -> List[Dict]:
    """
    STEP 3: Read YOLO format annotation file
    
    YOLO Format (normalized coordinates):
        class_id x_center y_center width height
        
    Example:
        0 0.523 0.421 0.634 0.721  # Disease at center, 63.4% width, 72.1% height
    
    Args:
        label_path: Path to .txt annotation file
        
    Returns:
        List of dicts with keys: 'class_id', 'x_center', 'y_center', 'width', 'height'
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            values = line.split()
            
            # YOLO must have exactly 5 values
            if len(values) != 5:
                print(f"⚠️  Invalid annotation at {label_path}:{line_num}")
                continue
            
            try:
                class_id, x_center, y_center, width, height = map(float, values)
                
                # Validate ranges
                assert 0 <= class_id < NUM_CLASSES, f"Invalid class_id: {class_id}"
                assert 0 <= x_center <= 1, f"Invalid x_center: {x_center}"
                assert 0 <= y_center <= 1, f"Invalid y_center: {y_center}"
                assert 0 <= width <= 1, f"Invalid width: {width}"
                assert 0 <= height <= 1, f"Invalid height: {height}"
                
                annotations.append({
                    'class_id': int(class_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                })
            except (ValueError, AssertionError) as e:
                print(f"⚠️  Error parsing line {line_num} in {label_path}: {e}")
    
    return annotations


# ============================================================================
# STEP 4: DATA AUGMENTATION
# ============================================================================

class AugmentationPipeline:
    """
    STEP 4: Augment images and maintain YOLO annotations
    
    Augmentations (with probability):
        - Horizontal flip (50%)
        - Vertical flip (30%)
        - Rotation ±15° (30%)
        - Brightness/contrast adjustment (50%)
        - Mosaic augmentation (combines 4 images)
    
    Why augment?
        - Increase effective dataset size
        - Teach model robustness to variations
        - Prevent overfitting
        - Simulate real-world lighting/angle changes
    """
    
    def __init__(self, image_size: int = 480, prob: float = 0.5):
        """
        Args:
            image_size: Target image size (square)
            prob: Probability of applying each augmentation
        """
        self.image_size = image_size
        self.prob = prob
    
    def __call__(self, image: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """
        Apply augmentations to image and update annotations
        
        Args:
            image: PIL Image
            annotations: List of YOLO annotations
            
        Returns:
            (augmented_image, updated_annotations)
        """
        # Resize to target size first
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            for ann in annotations:
                ann['x_center'] = 1.0 - ann['x_center']
        
        # Vertical flip
        if random.random() < 0.3:
            image = TF.vflip(image)
            for ann in annotations:
                ann['y_center'] = 1.0 - ann['y_center']
        
        # Rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, expand=False, fill=128)
        
        # Brightness/Contrast
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.85, 1.15)
            image = TF.adjust_brightness(image, brightness_factor)
            
            contrast_factor = random.uniform(0.85, 1.15)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, annotations


# ============================================================================
# STEP 5: PYTORCH DATASET CLASS
# ============================================================================

class TomatoDataset(Dataset):
    """
    STEP 5: PyTorch Dataset for YOLO object detection
    
    Loads images and their bounding box annotations.
    
    Returns:
        image: Tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE)
        targets: List of [class_id, x_center, y_center, width, height] (normalized)
    
    Pipeline:
        Raw image → Resize → Augment → Normalize → Tensor
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        image_size: int = IMAGE_SIZE,
        augment: bool = True
    ):
        """
        Args:
            image_dir: Path to images folder
            label_dir: Path to labels folder
            image_size: Target image size
            augment: Whether to apply augmentations (True for train, False for val)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.augment = augment
        
        # Collect all image files
        self.image_files = get_image_files(image_dir)
        
        # Augmentation pipeline
        self.augmentor = AugmentationPipeline(image_size) if augment else None
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
        print(f"Loaded {len(self.image_files)} images from {image_dir}")
    
    def __len__(self) -> int:
        """Return number of images in dataset"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            image: (C, H, W) tensor normalized
            targets: (N, 5) tensor [class_id, x_center, y_center, width, height]
                     N = number of objects in image
        """
        image_path = self.image_files[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            # Return black image on error
            image = Image.new("RGB", (self.image_size, self.image_size))
        
        # Load annotations
        annotations = read_yolo_label(label_path)
        
        # Augment (only for training)
        if self.augment and self.augmentor:
            image, annotations = self.augmentor(image, annotations)
        else:
            # Just resize for validation
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert annotations to tensor
        if annotations:
            targets = torch.tensor([
                [ann['class_id'], ann['x_center'], ann['y_center'], ann['width'], ann['height']]
                for ann in annotations
            ], dtype=torch.float32)
        else:
            # Image with no objects (should be rare)
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        # Normalize image
        image = self.normalize(image)
        
        return image, targets


# ============================================================================
# STEP 6: CREATE DATALOADERS
# ============================================================================

def create_dataloaders(
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    STEP 6: Create train and validation DataLoaders
    
    Args:
        batch_size: Images per batch
        num_workers: Parallel data loading threads
        augment: Whether to augment training data
        
    Returns:
        (train_loader, val_loader)
    """
    print("\n" + "="*70)
    print("STEP 6: Creating DataLoaders")
    print("="*70)
    
    # Training dataset (with augmentation)
    train_dataset = TomatoDataset(
        image_dir=IMAGE_TRAIN,
        label_dir=LABEL_TRAIN,
        augment=True
    )
    
    # Validation dataset (no augmentation)
    val_dataset = TomatoDataset(
        image_dir=IMAGE_VAL,
        label_dir=LABEL_VAL,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Custom batching
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches of {batch_size}")
    print(f"✓ Val loader: {len(val_loader)} batches of {batch_size}\n")
    
    return train_loader, val_loader


def collate_fn(batch):
    """
    Custom collate function for variable-sized bounding box lists
    
    Handles case where images have different numbers of objects.
    
    Returns:
        images: (B, C, H, W) tensor
        targets: List[Tensor] of shape (N_i, 5) for each image
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================================
# STEP 7: DATASET STATISTICS
# ============================================================================

def analyze_dataset():
    """
    STEP 7: Compute dataset statistics
    
    Reports:
        - Class distribution
        - Image sizes
        - Bounding box statistics
        - Objects per image
    """
    print("\n" + "="*70)
    print("STEP 7: Analyzing Dataset Statistics")
    print("="*70)
    
    # Class distribution
    class_counts = Counter()
    total_boxes = 0
    box_areas = []
    object_counts = []
    
    for label_file in LABEL_TRAIN.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        object_counts.append(len(lines))
        total_boxes += len(lines)
        
        for line in lines:
            values = line.split()
            if len(values) == 5:
                class_id = int(values[0])
                class_counts[class_id] += 1
                
                # Box area (normalized)
                width, height = float(values[3]), float(values[4])
                box_areas.append(width * height)
    
    # Print statistics
    print("\n📊 Class Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        pct = 100 * count / total_boxes
        print(f"  {class_id:2d} {CLASS_NAMES[class_id]:30s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n📊 Objects per Image:")
    print(f"  Total images: {len(object_counts)}")
    print(f"  Total objects: {total_boxes}")
    print(f"  Avg objects/image: {np.mean(object_counts):.2f}")
    print(f"  Max objects: {max(object_counts)}")
    print(f"  Min objects: {min(object_counts)}")
    
    if box_areas:
        print(f"\n📊 Bounding Box Sizes (normalized):")
        print(f"  Avg area: {np.mean(box_areas):.4f}")
        print(f"  Min area: {np.min(box_areas):.4f}")
        print(f"  Max area: {np.max(box_areas):.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" +  "="*70)
    print("🍅 TOMATO DISEASE DETECTION - PREPROCESSING")
    print("="*70)
    
    # Validate structure
    check_dataset_structure()
    
    # Analyze statistics
    analyze_dataset()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(batch_size=8, augment=True)
    
    # Test loading a batch
    print("\n" + "="*70)
    print("STEP 8: Testing DataLoader")
    print("="*70)
    
    images, targets = next(iter(train_loader))
    print(f"✓ Batch of images: {images.shape}")
    print(f"✓ Number of images in batch: {len(targets)}")
    
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            print(f"  Image {i}: {target.shape[0]} objects")
            print(f"    Classes: {target[:, 0].unique().int().tolist()}")
    
    print("\n✓ Preprocessing complete! Ready to train.\n")
