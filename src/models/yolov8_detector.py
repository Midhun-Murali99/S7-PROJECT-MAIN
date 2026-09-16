"""
=============================================================================
OBJECT DETECTION MODEL - YOLOv8-inspired Architecture
=============================================================================

This module implements a YOLOv8-style object detection model optimized for
tomato disease detection.

Architecture:
    Input → Backbone (CSPDarknet) → Neck (FPN) → Detection Head → Output
    
    Backbone: Extracts multi-scale features from the image
    Neck: Combines features from different scales (small to large)
    Head: Predicts bounding boxes & class probabilities
    
Key Concepts:
    - Multi-scale detection: Detects objects at 3 different scales
    - Anchor-free: Directly predicts box centers instead of using anchors
    - Efficient: Designed for real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ConvBlock(nn.Module):
    """
    Standard Conv → BatchNorm → Activation block
    
    Why BatchNorm?
        - Normalizes neuron inputs
        - Allows higher learning rates
        - Acts as regularization
        - Accelerates training
    
    Why SiLU activation?
        - Smooth activation function
        - Better gradients than ReLU
        - Used in YOLOv8
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual Block = ConvBlock → ConvBlock + Skip Connection
    
    Why skip connections?
        - Allows training deeper networks
        - Gradients flow directly through skip
        - Easier optimization
        - Reduces vanishing gradient problem
    
    Example:
        x → Conv1 → Conv2 → + → Output
        ↑_______________________↓
        (skip connection - identity)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels, 3, 1)
        self.cv2 = ConvBlock(out_channels, out_channels, 3, 1)
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class BottleneckBlock(nn.Module):
    """
    Bottleneck = ConvBlock(1x1) → ConvBlock(3x3) → ConvBlock(1x1) + Skip
    
    Why this pattern?
        - 1×1 reduces dimensions (cheaper computation)
        - 3×3 extracts features (main computation)
        - 1×1 expands back (matches skip connection)
        - Efficient: capture features with fewer parameters
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden, 1, 1, 0)
        self.cv2 = ConvBlock(hidden, out_channels, 3, 1, 1)
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x))


# ============================================================================
# BACKBONE: CSPDarknet
# ============================================================================

class CSPLayer(nn.Module):
    """
    CSP (Cross Stage Partial) Layer
    
    Divides the channel dimension, processes each path separately, then merges.
    
    Why CSP?
        - Reduces computation while preserving accuracy
        - Used in YOLOv4/v5/v8
        - Balances feature richness with efficiency
    
    Structure:
        Input (c channels)
            ↓
        Split into 2 paths (c/2 channels each)
            ↓
        Path 1: Conv → Conv + Skip
        Path 2: Conv
            ↓
        Concatenate → Conv → Output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True
    ):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden, 1, 1, 0)
        self.cv2 = ConvBlock(in_channels, hidden, 1, 1, 0)
        self.cv3 = ConvBlock(2 * hidden, out_channels, 1, 1, 0)
        
        self.blocks = nn.Sequential(
            *[BottleneckBlock(hidden, hidden) for _ in range(num_blocks)]
        )
        self.shortcut = shortcut
    
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.blocks(self.cv2(x))
        
        if self.shortcut:
            y2 = y1 + y2
        
        return self.cv3(torch.cat([y1, y2], 1))


class Backbone(nn.Module):
    """
    CSPDarknet Backbone - Feature Extractor
    
    Extracts features at 3 scales (8x, 16x, 32x downsampling):
    
    Input Image (480×480)
        ↓ Conv (stride=2) → 240×240
        ↓ CSP blocks
        ↓ Conv (stride=2) → 120×120  ← P3 (small object features)
        ↓ CSP blocks
        ↓ Conv (stride=2) → 60×60   ← P4 (medium object features)
        ↓ CSP blocks
        ↓ Conv (stride=2) → 30×30   ← P5 (large object features)
        ↓ CSP blocks
    
    Returns three feature maps at different resolutions.
    Why multiple scales?
        - Small objects visible at high resolution
        - Large objects visible at low resolution
        - Multi-scale detection ensures all sizes detected
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        # Initial convolution: 480×480 → 240×240
        self.stem = ConvBlock(in_channels, base_channels, 3, 2)
        
        # P1: 240×240
        self.dark2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, num_blocks=1)
        )
        
        # P2: 120×120 (small object features)
        self.dark3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, num_blocks=2)
        )
        
        # P3: 60×60 (medium object features)
        self.dark4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, num_blocks=2)
        )
        
        # P4: 30×30 (large object features)
        self.dark5 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, 3, 2),
            CSPLayer(base_channels * 16, base_channels * 16, num_blocks=1)
        )
    
    def forward(self, x):
        """
        Returns:
            p3: Small object features (120×120)
            p4: Medium object features (60×60)
            p5: Large object features (30×30)
        """
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        
        return p3, p4, p5


# ============================================================================
# NECK: Feature Pyramid Network (FPN)
# ============================================================================

class FPN(nn.Module):
    """
    Feature Pyramid Network (Neck) - Combines Multi-Scale Features
    
    Problem: 
        - Backbone gives features at 3 scales
        - Large objects encoded in small feature maps (few pixels)
        - Small objects lost in large feature maps
    
    Solution (FPN):
        - Upsample low-resolution features
        - Combine with high-resolution features
        - Create feature maps with rich multi-scale information
    
    Structure:
                    p5 (30×30)
                        ↓ (1×1 conv to reduce channels)
                        ↓ (upsample 2×)
                        ↓ (concat with p4)
                    p4_merged (60×60)
                        ↓ (1×1 conv to reduce channels)
                        ↓ (upsample 2×)
                        ↓ (concat with p3)
                    p3_merged (120×120)
    
    Result: Three feature maps combining information from all scales
    
    Why this works?
        - High-res features good for detecting small objects
        - Low-res features provide semantic information
        - Combination of both → robust multi-scale detection
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        # Reduce channel dimensions via 1×1 convolutions
        self.reduce_layer1 = ConvBlock(base_channels * 16, base_channels * 8, 1, 1, 0)
        self.reduce_layer2 = ConvBlock(base_channels * 8, base_channels * 4, 1, 1, 0)
        
        # Merge layers after concatenation
        self.merge_layer1 = CSPLayer(base_channels * 16, base_channels * 8, num_blocks=1)
        self.merge_layer2 = CSPLayer(base_channels * 8, base_channels * 4, num_blocks=1)
        self.merge_layer3 = CSPLayer(base_channels * 8, base_channels * 8, num_blocks=1)
    
    def forward(self, p3, p4, p5):
        """
        Args:
            p3: 120×120 features (64×64 channels)
            p4: 60×60 features (128×128 channels)
            p5: 30×30 features (256×256 channels)
        
        Returns:
            out3: 120×120 merged features
            out4: 60×60 merged features
            out5: 30×30 merged features
        """
        # Top-down: reduce p5 and upsample
        p5_reduced = self.reduce_layer1(p5)
        p4_upsample = F.interpolate(p5_reduced, scale_factor=2, mode='nearest')
        p4_merged = torch.cat([p4_upsample, p4], dim=1)
        p4_processed = self.merge_layer1(p4_merged)
        
        # Continue: reduce p4 and upsample
        p4_reduced = self.reduce_layer2(p4_processed)
        p3_upsample = F.interpolate(p4_reduced, scale_factor=2, mode='nearest')
        p3_merged = torch.cat([p3_upsample, p3], dim=1)
        p3_processed = self.merge_layer2(p3_merged)
        
        # Bottom-up: downample p3
        p3_downsample = nn.MaxPool2d(2, 2)(p3_processed)
        p4_cat = torch.cat([p3_downsample, p4_processed], dim=1)
        p4_final = self.merge_layer3(p4_cat)
        
        # Downsample p4
        p4_downsample = nn.MaxPool2d(2, 2)(p4_final)
        p5_cat = torch.cat([p4_downsample, p5], dim=1)
        p5_final = self.merge_layer1(p5_cat)
        
        return p3_processed, p4_final, p5_final


# ============================================================================
# DETECTION HEAD
# ============================================================================

class DetectionHead(nn.Module):
    """
    Detection Head - Predicts Bounding Boxes & Classes
    
    Input: Multi-scale feature maps (p3, p4, p5)
    Output: Predictions for each scale
    
    For each position in feature map:
        - 4 values: x_center, y_center, width, height (bbox regression)
        - 1 value: objectness score (is there an object?)
        - 10 values: class probabilities (which disease?)
        Total: 15 values per position
    
    Example (p3 at 120×120):
        Input: (B, 64, 120, 120)
        Processing: Conv layers to predict
        Output: (B, 120, 120, 15)  ← 15 = 4 bbox + 1 obj + 10 class
    
    During inference:
        - Apply softmax to class logits
        - Filter by objectness threshold
        - Apply NMS (non-maximum suppression) to remove duplicates
    """
    
    def __init__(self, base_channels: int = 64, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.pred_channels = 4 + 1 + num_classes  # bbox + objectness + classes
        
        # Prediction heads for each scale
        self.p3_head = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, 3),
            nn.Conv2d(base_channels * 8, self.pred_channels, 1)
        )
        
        self.p4_head = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, 3),
            nn.Conv2d(base_channels * 16, self.pred_channels, 1)
        )
        
        self.p5_head = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 32, 3),
            nn.Conv2d(base_channels * 32, self.pred_channels, 1)
        )
    
    def forward(self, p3, p4, p5):
        """
        Returns:
            out3: (B, H/4, W/4, 15) predictions
            out4: (B, H/8, W/8, 15) predictions
            out5: (B, H/16, W/16, 15) predictions
        """
        return self.p3_head(p3), self.p4_head(p4), self.p5_head(p5)


# ============================================================================
# FULL DETECTION MODEL
# ============================================================================

class YOLOv8ObjectDetector(nn.Module):
    """
    Complete YOLOv8-style Object Detection Model
    
    Pipeline:
        Image → Backbone → FPN → Head → Predictions → Post-processing
    
    Components:
        1. Backbone: Extract features at multiple scales
        2. Neck (FPN): Combine multi-scale features
        3. Head: Predict boxes and classes
    
    Input:
        (B, 3, 480, 480) batch of images
    
    Output (training):
        List of raw predictions at 3 scales
    
    Output (inference):
        List[Dict] with keys: box, confidence, class_id
    """
    
    def __init__(self, num_classes: int = 10, base_channels: int = 64):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = Backbone(3, base_channels)
        self.neck = FPN(base_channels)
        self.head = DetectionHead(base_channels, num_classes)
    
    def forward(self, x):
        """
        Forward pass through entire model
        
        Args:
            x: (B, 3, 480, 480) batch of images
        
        Returns:
            Tuple of three predictions:
            - out3: (B, 120, 120, 15) small object predictions
            - out4: (B, 60, 60, 15) medium object predictions
            - out5: (B, 30, 30, 15) large object predictions
        """
        # Backbone: extract multi-scale features
        p3, p4, p5 = self.backbone(x)
        
        # Neck: combine features
        p3_fpn, p4_fpn, p5_fpn = self.neck(p3, p4, p5)
        
        # Head: predict boxes and classes
        out3, out4, out5 = self.head(p3_fpn, p4_fpn, p5_fpn)
        
        return out3, out4, out5
    
    def count_parameters(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class YOLOLoss(nn.Module):
    """
    YOLOv8 Loss Function
    
    Multi-task loss combines three components:
    
    1. Box Loss (DIoU loss):
        - Measures how well predicted box matches ground truth
        - DIoU considers distance between centers and aspect ratio
        - Better than simple L2 loss
    
    2. Objectness Loss:
        - Binary classification: is there an object at this position?
        - Uses binary cross-entropy
    
    3. Class Loss:
        - Multi-class classification: which disease?
        - Uses cross-entropy with softmax
    
    Total Loss = λ_box × Box_Loss + λ_obj × Obj_Loss + λ_cls × Class_Loss
    
    λ values (hyperparameters):
        - λ_box = 7.5 (emphasize accurate localization)
        - λ_obj = 1.0 (standard weight)
        - λ_cls = 0.5 (classes less important than boxes)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.bce_pos = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0])
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        
        # Loss weights
        self.lambda_box = 7.5
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
    
    def forward(self, predictions, targets):
        """
        Compute loss between predictions and ground truth
        
        Args:
            predictions: Tuple of 3 scale outputs
            targets: List of target tensors (one per image)
        
        Returns:
            loss: Scalar loss value
        """
        # TODO: Implement loss calculation
        # For now, dummy implementation
        return torch.tensor(0.0, requires_grad=True).to(predictions[0].device)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_model(num_classes: int = 10) -> YOLOv8ObjectDetector:
    """Create and return model instance"""
    model = YOLOv8ObjectDetector(num_classes=num_classes)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    print("\n" + "="*70)
    print("🍅 YOLOV8 OBJECT DETECTION MODEL")
    print("="*70 + "\n")
    
    model = create_model(num_classes=10)
    
    # Test forward pass
    x = torch.randn(2, 3, 480, 480)  # Batch of 2 images
    out3, out4, out5 = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shapes:")
    print(f"  P3 (120×120): {out3.shape}")
    print(f"  P4 (60×60):   {out4.shape}")
    print(f"  P5 (30×30):   {out5.shape}")
    
    print(f"\n✓ Model test passed!")
