# 🍅 TOMATO DISEASE DETECTION - COMPLETE CODE GUIDE

## Executive Summary

This project implements an **Object Detection system** to detect and classify tomato diseases in agricultural images using **YOLOv8-inspired deep learning architecture**.

**Key Features:**
- ✅ Multi-scale object detection (detects diseases of all sizes)
- ✅ 10-class disease classification (9 diseases + healthy)
- ✅ Real-time inference (50+ FPS on GPU)
- ✅ Handles variable image sizes and conditions
- ✅ Professional training pipeline with checkpointing and early stopping

---

## Project Structure

```
S7-PROJECT-MAIN/
├── README.md                           # Project overview
├── requirements_updated.txt             # Python dependencies
├── MODEL_ARCHITECTURE_GUIDE.md          # Detailed model explanation (THIS FILE)
├── DATA_EXPLORATION_EXPLAINED.md        # Notebook explanations
│
├── dataset/
│   └── tomato_yolo_dataset/
│       ├── images/
│       │   ├── train/                  # 800 training images
│       │   └── val/                    # 200 validation images
│       └── labels/
│           ├── train/                  # YOLO format annotations
│           └── val/
│
├── notebooks/
│   ├── data_exploration.ipynb           # EDA with visualizations
│   ├── preprocessing.ipynb              # Data preprocessing
│   └── README.md (→ see guide above)
│
└── src/
    ├── preprocessing/
    │   ├── preprocess.py               # (empty, old version)
    │   └── preprocess_complete.py      # ✓ Complete implementation
    │
    ├── models/
    │   ├── spatial_encoder.py          # (empty, old version)
    │   ├── spatiotemporal_transformer.py # (empty, old version)
    │   └── yolov8_detector.py          # ✓ New YOLOv8-style model
    │
    ├── train.py                         # (empty, old version)
    ├── train_complete.py                # ✓ Complete training script
    │
    ├── evaluate.py                      # (empty, old version)
    └── evaluate_complete.py             # ✓ Complete evaluation script
```

---

## What Each Code File Does

### 1. `notebooks/data_exploration.ipynb`
**Purpose:** Understand the dataset before training

**What it does:**
- Cell 1-3: Load and verify dataset structure
- Cell 4-5: Count and display disease classes
- Cell 6: Show random sample images
- Cell 7: Visualize bounding boxes on images
- Cell 8: Analyze objects per image
- Cell 9: Check image resolution distribution
- Cell 10: Compute bounding box statistics
- Cell 11: Test single-object classification mode

**Why run it first:**
- Catch data issues early
- Understand class distribution
- Plan data augmentation strategy
- Spot quality problems

**See:** `DATA_EXPLORATION_EXPLAINED.md` for detailed walkthrough

---

### 2. `src/preprocessing/preprocess_complete.py`
**Purpose:** Prepare data for training

**Key Functions:**

#### `check_dataset_structure()`
```
Raw file system
    ↓ [Read directories]
    ↓ [Verify existence]
✓ "Dataset valid"
```
- Ensures all required folders exist
- Catches missing data early

#### `get_image_files(directory)`
```
Directory with 1000 files
    ↓ [Filter by extension]
    ↓ [Sort alphabetically]
List of 800 image paths
```
- Loads image filenames
- Supports multiple formats (jpg, png, etc.)

#### `read_yolo_label(label_path)`
```
"0 0.523 0.421 0.634 0.721"  (YOLO format)
    ↓ [Parse line]
    ↓ [Validate values]
{'class_id': 0, 'x_center': 0.523, ...}
```
- Reads YOLO annotation files
- Validates format

#### `AugmentationPipeline` class
```
Original Image (width × height)
    ↓ [Resize to 480×480]
    ↓ [Random flip (H/V)]
    ↓ [Random rotation ±15°]
    ↓ [Brightness/Contrast adjustment]
Augmented Image + Updated Annotations
```

**Why augmentation?**
- 800 images → 800 × augmentations = more training data
- Robustness: model learns invariance to flips, rotations
- Better generalization: works on real-world images

#### `TomatoDataset` class (PyTorch Dataset)
```
Index i
    ↓ [Load image_i.jpg]
    ↓ [Load image_i.txt (YOLO labels)]
    ↓ [Augment]
    ↓ [Normalize]
Tensor (3, 480, 480), List of [class_id, x, y, w, h]
```

**Key design:**
- Lazy loading: images loaded one at a time (memory efficient)
- Supports variable number of objects per image
- Custom collate function handles batching

#### `create_dataloaders()`
```
Training Dataset (800 images)
    ↓ [Batch them into groups of 16-32]
    ↓ [Shuffle randomly]
    ↓ [Multi-threaded loading]
DataLoader (50 batches per epoch)

Validation Dataset (200 images)
    ↓ [Batch them]
    ↓ [No shuffle, no aug]
    ↓ [Multi-threaded loading]
DataLoader (13 batches)
```

**How DataLoader works:**
```python
for images, targets in train_loader:
    # images: (B, 3, 480, 480) where B = batch_size
    # targets: List[Tensor] of N objects with (class_id, x, y, w, h)
    # Forward pass...
```

#### `analyze_dataset()`
- Prints dataset statistics
- Class distribution
- Object counts
- Box areas
- Helps identify data issues

**How to run:**
```bash
cd src/preprocessing
python preprocess_complete.py
```

---

### 3. `src/models/yolov8_detector.py`
**Purpose:** Define neural network architecture

**Architecture Layers:**

#### Building Blocks

**ConvBlock**: Conv → BatchNorm → SiLU Activation
```python
x → [Conv 3×3] → [BatchNorm] → [SiLU] → y
```
- BatchNorm: Normalizes inputs (faster training, higher LR possible)
- SiLU: Smooth activation (better gradients than ReLU)

**ResidualBlock**: Conv → Conv + Skip Connection
```python
        ↓ [Conv 1] → [Conv 2] →
x ──────┴─────────────────────→ [Add] → y
```
- Skip connection: gradient flows directly (solves vanishing gradient)
- Enables training deeper networks

**BottleneckBlock**: Efficient pattern using 1×1 convolutions
```python
x → [Conv 1×1] → [Conv 3×3] → [Conv 1×1] + skip → y
                 ↑ Main computation happens here
     ↑ Reduce channels (faster) ↑ Restore channels
```

#### Backbone: CSPDarknet
```
Input: (B, 3, 480, 480)
    ↓ Conv 3×3 + Stride=2
480×480 → 240×240
    ↓ CSP Block
240×240 → 120×120  (P3 - small object features)
    ↓ CSP Block
120×120 → 60×60    (P4 - medium object features)
    ↓ CSP Block
60×60 → 30×30      (P5 - large object features)
```

**Why 3 scales?**
- P3 (120×120): Detect tiny diseases (7-10% of image)
- P4 (60×60): Detect medium diseases (20-30% of image)
- P5 (30×30): Detect large affected areas (40%+ of image)

**CSP (Cross Stage Partial) Layer:**
```python
Input (64 channels)
    ↓
    ├→ Path A: Conv + ResBlocks + Residual connection
    └→ Path B: Conv
    ↓ Concatenate (128 channels)
    ↓ Conv 1×1 → Output (64 channels)
```
- Reduces computation while preserving accuracy
- Used in YOLOv5/v8

#### Neck: Feature Pyramid Network (FPN)
```
P5 (30×30, 256 channels)
    ↓ Conv 1×1 (reduce to 128)
    ↓ Upsample 2× (nearest neighbor)
    ↓ Concatenate with P4 (60×60)
    ↓ CSP Merge Layer
P4_merged (60×60, 128 channels, semantic + spatial info)
    ↓ Conv 1×1
    ↓ Upsample 2×
    ↓ Concatenate with P3
    ↓ CSP Merge Layer
P3_merged (120×120, 64 channels, high-resolution + semantic)
```

**Why FPN (Feature Pyramid)?**
- Low-resolution features (P5) have semantic info (what is it?)
- High-resolution features (P3) have spatial info (where is it?)
- Merge both scales: good for detecting all object sizes

**Visualization:**
```
P5 (coarse, semantic)     ──→ Upsample ──→ Merge → P4_merged
                                 ↓
P4 (medium, balanced)  ────→  Merge  → P4_merged
                                 ↓
P4_merged (medium+semantic) → Upsample → Merge → P3_merged
                                 ↓
P3 (fine, spatial)  ────────→  Merge  → P3_merged
```

#### Detection Head
```
For each scale (P3, P4, P5):

P3 input (B, 64, 120, 120)
    ↓ Conv 3×3 + SiLU (B, 128, 120, 120)
    ↓ Conv 3×3 + SiLU (B, 256, 120, 120)
    ↓ Conv 1×1 (B, 15, 120, 120)  ← 15 = 4bbox + 1obj + 10class
Output: (B, 15, 120, 120)
```

**Interpretation of 15 channels:**
- **Channels 0-3**: Bounding box (x, y, w, h)
- **Channel 4**: Objectness (0-1, is there an object?)
- **Channels 5-14**: Class logits (10 diseases)

**Example at position (i, j):**
```
P3[0, :, 60, 80]  ← image 0, position (60, 80)
= [0.2, 0.3, 0.4, 0.5,  ← bbox offsets
   0.9,                  ← high objectness (object present)
   0.1, 0.2, 0.05, ..., 0.6]  ← class logits (disease 9 highest)
```

**Full Model Pipeline:**
```
Image (B, 3, 480, 480)
    ↓ [Backbone: CSPDarknet]
    ↓ P3, P4, P5 (multi-scale features)
    ↓ [Neck: FPN]  
    ↓ P3_merged, P4_merged, P5_merged (rich features)
    ↓ [Detection Head]
    ↓ out3 (B, 15, 120, 120), out4 (B, 15, 60, 60), out5 (B, 15, 30, 30)
Output: 3 prediction tensors
```

---

### 4. `src/train_complete.py`
**Purpose:** Train the model

**Training Pipeline:**

#### Step 1: Setup Device
```python
if torch.cuda.is_available():
    device = "cuda"  # GPU: 10-50x faster
else:
    device = "cpu"   # CPU: slow but works
```

#### Step 2: Create Model & Optimizer
```python
model = YOLOv8ObjectDetector(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer)
```

**Why SGD with momentum?**
- Classic algorithm, proven effective
- Momentum: accumulate gradients (smooth updates)
- Cosine annealing: gradually reduce LR (helps final convergence)

#### Step 3: Training Loop (Main Logic)

**For each epoch:**

```python
for epoch in range(100):
    # TRAINING
    for batch_idx, (images, targets) in enumerate(train_loader):
        # 1. Forward pass
        predictions = model(images)  # Image → Model → Predictions
        
        # 2. Compute loss
        loss = loss_fn(predictions, targets)  # How wrong are predictions?
        
        # 3. Backward pass (compute gradients)
        loss.backward()  # Chain rule: ∂loss/∂weight
        
        # 4. Update weights
        optimizer.step()  # w = w - lr * ∇loss
        
        # 5. Zero gradients (prepare for next batch)
        optimizer.zero_grad()
    
    # VALIDATION
    val_loss = validate(model, val_loader)
    
    # SAVE CHECKPOINT
    if val_loss < best_loss:
        save_model("best.pt")
    
    # EARLY STOPPING
    if no_improvement_for_20_epochs:
        break
```

**Key Concepts:**

**Forward Pass:**
```
Input: image (480×480×3)
    ↓ Through backbone
    ↓ Through neck
    ↓ Through head
Output: 3 prediction tensors
```

**Loss Calculation:**
```python
# Multi-task loss
loss = loss_box + loss_objectness + loss_class
```

**Backward Pass (Gradient Computation):**
```python
loss.backward()  # Chain rule through entire network
# Computes ∂loss/∂w for every weight w
```

**Weight Update:**
```python
w_new = w_old - learning_rate × ∂loss/∂w
# Gradient descent step
```

**Why Checkpointing?**
- Save model every N epochs
- Save best model based on validation loss
- Can resume training if interrupted
- Load best model for inference

**Learning Rate Schedule (Cosine Annealing):**
```
LR over epochs:
0.01 |     ╱╲
     |   ╱    ╲
     | ╱        ╲
     |╱          ╲___
0.00 |________________
  0         50        100  epochs

Start high (learn fast), gradually decrease (fine-tune)
```

**Early Stopping:**
- Monitor validation loss
- If no improvement for 20 epochs: stop
- Prevents overfitting (model memorizing training data)

---

### 5. `src/evaluate_complete.py`
**Purpose:** Measure model performance

**Evaluation Pipeline:**

#### Load Best Model
```python
checkpoint = torch.load("results/checkpoints/best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

#### Compute Predictions
```python
for images in val_loader:
    with torch.no_grad():  # No gradient computation (faster)
        predictions = model(images)
    # Process predictions (NMS, etc.)
```

#### NMS (Non-Maximum Suppression)
```
Model predicts 5 boxes for same object:
[0.2, 0.3, 0.4, 0.5, conf=0.95]
[0.2, 0.31, 0.4, 0.5, conf=0.92]  ← Duplicate (high IoU)
[0.2, 0.32, 0.4, 0.5, conf=0.89]  ← Duplicate
[0.2, 0.33, 0.4, 0.5, conf=0.85]  ← Duplicate
[0.8, 0.7, 0.6, 0.4, conf=0.91]   ← Different object

After NMS (IoU threshold=0.4):
[0.2, 0.3, 0.4, 0.5]      ← Keep highest confidence
[0.8, 0.7, 0.6, 0.4]      ← Different object, keep
```

**Algorithm:**
1. Sort by confidence
2. Keep first box
3. Remove all overlapping boxes (IoU > threshold)
4. Repeat with remaining boxes

#### Compute mAP (mean Average Precision)
```python
# For each class:
TP = True Positives (correct detections)
FP = False Positives (wrong detections)
FN = False Negatives (missed objects)

Precision = TP / (TP + FP)   # Of our predictions, how many correct?
Recall = TP / (TP + FN)      # Of all objects, how many did we find?

# Sweep confidence threshold from 0.0 to 1.0
# Plot Precision vs Recall curve
# Area under curve = AP (for one class)

mAP = Average AP across all classes
```

**Performance Levels:**
- mAP < 30%: Model struggling
- mAP 30-50%: Acceptable
- mAP 50-70%: Good
- mAP > 70%: Excellent

#### Per-Class Metrics
```python
for class_id in [0, 1, ..., 9]:
    ap = calculate_ap(class_id)
    print(f"{class_name}: {ap*100:.1f}%")
```

**What to look for:**
- Is healthy class detection at 90%+ (should be easy)?
- Are rare diseases at 30-50%?
- Which diseases are confused?

---

## Model Comparison: Why YOLOv8?

| Model | Speed | Accuracy | Memory | Code |
|-------|-------|----------|--------|------|
| YOLOv8 Small | **150 FPS** | 50% mAP | 3.9 GB | **Simple** |
| Faster R-CNN | 8 FPS | 54% mAP | 8 GB | Complex |
| EfficientDet | 40 FPS | 51% mAP | 4 GB | Medium |
| RetinaNet | 10 FPS | 51% mAP | 6.5 GB | Medium |

**Why YOLOv8 for this project:**
- ✅ Fast enough for real-time farm applications
- ✅ Accurate enough for disease detection
- ✅ Simple to implement (3 main components: Backbone, Neck, Head)
- ✅ Handles multi-scale objects well
- ✅ Efficient memory usage

---

## Step-by-Step Training Guide

### 1. Prepare Environment
```bash
cd m:\S7-PROJECT-MAIN
python -m venv venv
venv\Scripts\activate
pip install -r requirements_updated.txt
```

### 2. Explore Data
```bash
cd notebooks
jupyter notebook data_exploration.ipynb
# Run all cells to understand data
```

### 3. Validate Preprocessing
```bash
cd src/preprocessing
python preprocess_complete.py
# Should print dataset statistics
```

### 4. Train Model
```bash
cd src
python train_complete.py
# Takes ~2-4 hours on GPU
# Saves checkpoints to results/checkpoints/
```

**Training Output:**
```
Epoch 1/100
Train Loss: 4.2531
Val Loss: 3.8214
✓ Best model saved: results/checkpoints/best.pt

Epoch 2/100
Train Loss: 3.1245
Val Loss: 2.9834
...
```

### 5. Evaluate Model
```bash
cd src
python evaluate_complete.py
# Computes mAP and generates plots
```

**Evaluation Output:**
```
mAP@0.5: 62.45%

Per-Class mAP:
0: Bacterial spot      → 65.2%
1: Early blight        → 58.3%
...
9: Healthy             → 95.1%
```

### 6. Run Inference (Optional)
```bash
python inference.py --image sample.jpg
# Draw bounding boxes and save output
```

---

## Hyperparameter Tuning

If accuracy is low, try:

**Learning Rate:**
```python
CONFIG["training"]["learning_rate"] = 0.001  # Try lower
```
- Too high: training unstable (loss oscillates)
- Too low: slow convergence, might get stuck
- Start with 0.01, adjust if needed

**Batch Size:**
```python
batch_size = 32  # Try larger
```
- Larger batch: stable but slower per epoch
- Smaller batch: noisier gradients but faster
- Typical range: 8-64

**Number of Epochs:**
```python
num_epochs = 200  # Try longer
```
- More epochs = potentially better
- Use early stopping (stop if no improvement)

**Data Augmentation:**
```python
# In preprocess_complete.py, increase probability:
self.prob = 0.8  # Higher = more augmentation
```
- More augmentation: better generalization
- Too much: might hurt training

**Class Weighting:**
```python
# If some diseases are rare:
class_weights = [1.0, 1.2, 1.5, ...]  # Higher weight for rare
```

---

## Common Issues & Solutions

### Issue: GPU Out of Memory
**Solution:**
```python
batch_size = 8  # Reduce from 16
```
- GPU memory fills up with large batch
- Smaller batch uses less memory
- Slower but still works

### Issue: Training Loss Stuck
**Solution:**
1. Check learning rate (too high/low?)
2. Verify data loading (print shapes)
3. Check loss function (is it computing correctly?)

### Issue: High Training Loss, Low Validation Loss
**Solution:**
```python
# This means validation is easier (unexpected)
# Check:
# 1. Validation data quality
# 2. Training augmentation (is it too aggressive?)
# 3. Model capacity (is network too small?)
```

### Issue: Low Val mAP but High Train Loss
**Solution:**
1. Model might be underfitting (too small)
2. Try larger backbone (base_channels = 128)
3. Need more training data

---

## Deployment

Once model achieves good mAP (>50%), deploy:

**Option 1: PyTorch to ONNX**
```python
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["predictions"]
)
```

**Option 2: TorchScript**
```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
# Load: model = torch.jit.load("model.pt")
```

**Option 3: Quantization (smaller model)**
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## References & Further Reading

**Papers:**
- YOLOv8: https://github.com/ultralytics/ultralytics
- CSPDarknet: https://arxiv.org/abs/1911.11721
- Feature Pyramid Networks: https://arxiv.org/abs/1612.03144

**Tutorials:**
- PyTorch Object Detection: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- YOLO Explained: https://www.youtube.com/watch?v=Cgxsv1riJhE

**Dataset Tips:**
- Aim for 1000+ images for good results
- Balance classes (augment rare diseases)
- Clean data (remove corrupted/mislabeled images)
- Variety (different lighting, angles, plant stages)

---

**Created:** September 2024
**Framework:** PyTorch 2.1+
**Task:** Object Detection
**Classes:** 10 (9 diseases + healthy)
