# Tomato Disease Detection - Model Architecture Guide

## Project Overview
This is an **Object Detection** task using YOLO format annotations to detect and classify tomato diseases in images.

**Dataset**: 10 classes (9 diseases + Healthy)
- Bacterial spot
- Early blight  
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- Healthy

---

## Current Architecture & Why It Needs Changes

### ❌ **Original Plan: Spatiotemporal Transformer**
The project mentions `spatiotemporal_transformer.py` which assumes:
- **Temporal sequences** (video frames over time)
- **Complex feature interactions**
- High computational cost

This is **OVERKILL** because:
- YOLO dataset has **single static images**, not videos
- Image-to-image inference doesn't need temporal information
- Transformer models are slow for real-time disease detection

### ✅ **Recommended Architecture: YOLOv8-based Object Detection**

We'll implement a **two-tier detection system**:

```
Input Image (480×480)
        ↓
   [YOLOv8 Backbone]  ← Feature extraction
        ↓
   [Neck (FPN)]       ← Multi-scale fusion
        ↓
   [Detection Head]   ← Predict boxes + class scores
        ↓
Output: [x, y, w, h, confidence, class_id]
```

#### Why YOLOv8?
1. **Speed**: Real-time inference (50+ FPS on GPU)
2. **Accuracy**: 53.7 mAP on COCO dataset
3. **Simplicity**: Just 3 lines of code to train
4. **Built for this**: Native YOLO format support
5. **Flexible**: Can use pretrained backbone from ImageNet

#### Architecture Details:

**Backbone (CSPDarknet-based)**:
- 3×3 convolutions with batch normalization
- Skip connections for gradient flow
- Depthwise separable convolutions (efficient)
- Output: Feature maps at 3 scales (8x, 16x, 32x downsampling)

**Neck (Pyramid Feature Network - PFN)**:
- Combines features from different scales
- Top-down pathway: 1×1 convolutions + upsampling
- Bottom-up pathway: concatenation + convolutions
- Purpose: Detect objects at all scales (small diseases on leaves, large affected areas)

**Detection Head**:
- Predicts at 3 scales for multi-scale object detection
- For each scale position: 
  - **4 bbox coords**: x_center, y_center, width, height (normalized)
  - **1 objectness**: P(object exists)
  - **10 class logits**: P(each disease class)
- Outputs: (batch, num_predictions, 15)

---

## Alternative Models Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **YOLOv8s** | 150+ FPS | 50.2 mAP | 3.9 GB | **Our Choice** ✓ |
| YOLOv8m | 100 FPS | 52.3 mAP | 6.2 GB | Better accuracy |
| Faster R-CNN | 8 FPS | 53.3 mAP | 8 GB | Highest accuracy |
| EfficientDet | 40 FPS | 51.4 mAP | 4.1 GB | Good balance |
| RetinaNet | 10 FPS | 50.9 mAP | 6.5 GB | Handle class imbalance |

**For this project: YOLOv8 Small is optimal** - Fast enough for real-time use, accurate enough for disease detection.

---

## Data Pipeline

```
Raw Image (variable size)
        ↓
[Resize to 480×480]  ← Standardize input
        ↓
[Augmentation]:
  - Random flip (H/V)
  - Random rotation (-15° to +15°)
  - Random brightness/contrast
  - Random mosaic augmentation
        ↓
[Normalize]:
  - RGB to values [0, 1]
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]  ← ImageNet normalization
        ↓
[PyTorch Tensor]
        ↓
Model Forward Pass
```

---

## Training Strategy

### Loss Function
Uses **multi-task loss**:
```
Total Loss = 
    L_box (Localization) +
    L_cls (Classification) +
    L_obj (Objectness)
```

Details:
- **Box Loss**: IoU-based (DIoU loss) - measures bounding box accuracy
- **Class Loss**: Cross-entropy - measures disease classification
- **Object Loss**: Binary cross-entropy - P(object exists)

### Hyperparameters
- **Optimizer**: SGD with momentum (0.937)
- **Learning rate**: 0.01 (cosine annealing schedule)
- **Batch size**: 16 (adjust based on GPU memory)
- **Epochs**: 100-200
- **Weight decay**: 0.0005 (L2 regularization)
- **Warm-up**: 3 epochs (gradually increase LR from 0)

### Training Data Strategy
- **Train/Val split**: 80/20 from your data
- **Class balancing**: Weighted loss for imbalanced classes
- **Early stopping**: Stop if val loss doesn't improve for 20 epochs
- **Model checkpointing**: Save best weights based on mAP metric

---

## Why This Architecture Works for Your Dataset

1. **Multi-scale Detection**: 
   - Small diseases visible on small parts of leaves → small anchor boxes
   - Large affected areas → large anchor boxes
   - FPN ensures all scales are handled

2. **One-stage Detection**: 
   - Direct bbox + class prediction (fast)
   - No separate region proposal step

3. **Robustness**:
   - Object at any position in image
   - Different lighting conditions (augmentation)
   - Variable leaf sizes and orientations

4. **Efficiency**:
   - Fast inference for real-time disease detection in farms
   - Low memory footprint (deployable on edge devices)

---

## Evaluation Metrics

We use **mAP (mean Average Precision)** - standard for object detection:

```
For each class:
  1. For varying confidence thresholds (0.5 to 0.95)
  2. Calculate: True Positives, False Positives, False Negatives
  3. Compute Precision = TP / (TP + FP)
  4. Compute Recall = TP / (TP + FN)
  5. Get Area Under Precision-Recall Curve
6. Average across all classes → mAP
```

Also report:
- **mAP@0.5**: IoU ≥ 0.5 (standard practice)
- **mAP@0.75**: IoU ≥ 0.75 (stricter)
- **Per-class mAP**: See which diseases are hardest to detect
- **Speed**: FPS (inference frames per second)

---

## Files Explanation

### `preprocess.py`
- Load images and YOLO labels
- Custom PyTorch Dataset class
- Data augmentation pipeline
- Create train/val DataLoaders

### `models.py` (replaces spatial_encoder.py + spatiotemporal_transformer.py)
- YOLOv8 backbone implementation
- Feature pyramid neck
- Detection head
- Loss calculation functions

### `train.py`
- Data loading
- Model initialization
- Training loop with validation
- Learning rate scheduling
- Model checkpointing

### `evaluate.py`
- Load trained model
- Compute mAP metric
- Visualize predictions on test set
- Per-class metrics

---

## Next Steps

1. Install dependencies: `pip install -r requirements-updated.txt`
2. Run preprocessing: `python src/preprocessing/preprocess.py` (validates data)
3. Train model: `python src/train.py`
4. Evaluate: `python src/evaluate.py --checkpoint best.pt`
5. Run inference: `python src/inference.py --image sample.jpg`
