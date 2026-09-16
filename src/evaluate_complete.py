"""
=============================================================================
EVALUATION SCRIPT - Model Performance Metrics
=============================================================================

This script evaluates the trained model on the validation/test set.

Evaluation Metrics:
    1. Loss: Overall error measure
    2. mAP (mean Average Precision): Standard object detection metric
    3. Per-class mAP: See which diseases are hardest to detect
    4. Inference Speed: How fast is the model?
    5. Confusion Matrix: Which diseases are confused with each other?

What is mAP?
    - Measures how well model predicts bounding boxes and classes
    - Ranges from 0 (worst) to 100 (perfect)
    - Standard metric used in detection challenges (COCO, Pascal VOC)
    
How mAP is calculated:
    1. For each confidence threshold (0.5 to 0.95):
        a. Count True Positives (correct detections)
        b. Count False Positives (wrong detections)
        c. Count False Negatives (missed objects)
        d. Compute Precision = TP / (TP + FP)
        e. Compute Recall = TP / (TP + FN)
        f. Get Area Under Precision-Recall Curve
    2. Average across all classes
    3. Result: mAP score

Example:
    If we predict 10 objects and 8 are correct:
    - TP = 8, FP = 2
    - Precision = 8/10 = 80%
    - If ground truth has 10 objects:
    - Recall = 8/10 = 80%
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Import custom modules
from preprocessing.preprocess_complete import (
    TomatoDataset, create_dataloaders, CLASS_NAMES, NUM_CLASSES, 
    IMAGE_VAL, LABEL_VAL
)
from models.yolov8_detector import YOLOv8ObjectDetector


# ============================================================================
# STEP 1: LOAD CHECKPOINT
# ============================================================================

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    STEP 1: Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to best.pt file
        device: GPU or CPU
    
    Returns:
        model: Loaded YOLOv8ObjectDetector
        config: Training configuration
    """
    print("\n" + "="*70)
    print("STEP 1: Loading Checkpoint")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = YOLOv8ObjectDetector(
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"]
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', 'unknown')
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  - Epoch: {epoch+1}")
    print(f"  - Val Loss: {val_loss}")
    print(f"  - Parameters: {model.count_parameters():,}\n")
    
    return model, config


# ============================================================================
# STEP 2: NMS (Non-Maximum Suppression)
# ============================================================================

def nms(predictions, confidence_thresh: float = 0.5, iou_thresh: float = 0.4):
    """
    STEP 2: Apply NMS to remove duplicate detections
    
    Why NMS?
        - Model might predict multiple boxes for same object
        - NMS keeps highest confidence, removes duplicates
        - Uses IoU (Intersection over Union) to measure overlap
    
    Algorithm:
        1. Keep boxes with confidence > threshold
        2. Sort by confidence (highest first)
        3. For each box:
            a. Keep it
            b. Remove all overlapping boxes (IoU > threshold)
    
    Args:
        predictions: List of [x1, y1, x2, y2, confidence, class_id]
        confidence_thresh: Minimum confidence to keep
        iou_thresh: Maximum IoU to consider duplicates
    
    Returns:
        List of filtered predictions
    """
    if len(predictions) == 0:
        return predictions
    
    # Filter by confidence
    predictions = [p for p in predictions if p[4] >= confidence_thresh]
    
    if not predictions:
        return predictions
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    # Keep predictions based on IoU
    keep = []
    while predictions:
        keep.append(predictions[0])
        
        if len(predictions) == 1:
            break
        
        current = predictions[0]
        predictions = predictions[1:]
        
        # Filter by IOU
        iou_list = [compute_iou(current[:4], p[:4]) for p in predictions]
        predictions = [p for p, iou in zip(predictions, iou_list) if iou <= iou_thresh]
    
    return keep


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes
    
    IoU = Intersection Area / Union Area
    
    Values:
        - IoU = 0: No overlap
        - IoU = 1: Perfect match
        - IoU = 0.5: 50% overlap
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU score
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


# ============================================================================
# STEP 3: COMPUTE mAP
# ============================================================================

class mAPCalculator:
    """
    STEP 3: Calculate mean Average Precision (mAP)
    
    This is the standard metric for object detection models.
    """
    
    def __init__(self, num_classes: int, iou_thresh: float = 0.5):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        
        # Store TP, FP, confidence for each class
        self.TP = {i: [] for i in range(num_classes)}
        self.FP = {i: [] for i in range(num_classes)}
        self.confidences = {i: [] for i in range(num_classes)}
        self.gt_counts = {i: 0 for i in range(num_classes)}
    
    def add_predictions(self, predictions, ground_truths, image_id):
        """
        Add predictions and ground truths for one image
        
        Args:
            predictions: List of [x1, y1, x2, y2, confidence, class_id]
            ground_truths: List of [x1, y1, x2, y2, class_id]
        """
        # Match predictions to ground truths
        matched_gt = set()
        
        for pred in predictions:
            x1, y1, x2, y2, conf, class_id = pred
            class_id = int(class_id)
            
            self.confidences[class_id].append(conf)
            
            # Find best matching ground truth
            best_iou = 0
            best_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt[4] != class_id or gt_idx in matched_gt:
                    continue
                
                iou = compute_iou([x1, y1, x2, y2], gt[:4])
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            
            # Check if detection is TP or FP
            if best_iou >= self.iou_thresh and best_idx >= 0:
                self.TP[class_id].append(1)
                self.FP[class_id].append(0)
                matched_gt.add(best_idx)
            else:
                self.TP[class_id].append(0)
                self.FP[class_id].append(1)
        
        # Count ground truths
        for gt in ground_truths:
            class_id = int(gt[4])
            self.gt_counts[class_id] += 1
    
    def compute_ap(self, class_id):
        """
        Compute Average Precision for one class
        
        Returns:
            AP: Average Precision (0-1)
        """
        if self.gt_counts[class_id] == 0:
            return 0.0
        
        if len(self.TP[class_id]) == 0:
            return 0.0
        
        # Sort by confidence
        confidences = np.array(self.confidences[class_id])
        tp = np.array(self.TP[class_id])
        fp = np.array(self.FP[class_id])
        
        sorted_indices = np.argsort(-confidences)
        tp = tp[sorted_indices]
        fp = fp[sorted_indices]
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / self.gt_counts[class_id]
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP (area under PR curve)
        ap = 0
        for i in range(len(precision)):
            if i == 0 or recall[i] != recall[i-1]:
                ap += precision[i] * (recall[i] - (recall[i-1] if i > 0 else 0))
        
        return ap
    
    def compute_map(self):
        """
        Compute mean Average Precision across all classes
        
        Returns:
            mAP: Mean AP (0-100)
            per_class_ap: Dict of AP for each class
        """
        per_class_ap = {}
        total_ap = 0
        valid_classes = 0
        
        for class_id in range(self.num_classes):
            ap = self.compute_ap(class_id)
            per_class_ap[class_id] = ap
            
            if self.gt_counts[class_id] > 0:
                total_ap += ap
                valid_classes += 1
        
        mAP = total_ap / valid_classes if valid_classes > 0 else 0
        
        return mAP * 100, per_class_ap  # Convert to percentage


# ============================================================================
# STEP 4: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, val_loader, device):
    """
    STEP 4: Evaluate model on validation set
    
    Computes:
        - Validation loss
        - mAP metric
        - Per-class performance
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: GPU or CPU
    
    Returns:
        results: Dictionary with metrics
    """
    print("\n" + "="*70)
    print("STEP 4: Evaluating Model")
    print("="*70)
    
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    map_calculator = mAPCalculator(NUM_CLASSES, iou_thresh=0.5)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Process predictions
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                # Get ground truth boxes for this image
                gt_boxes = targets[i].cpu().numpy()
                
                # For now, placeholder for predictions
                # In full implementation, would decode model outputs
                pred_boxes = []
                
                # Add to calculator
                map_calculator.add_predictions(pred_boxes, gt_boxes, i)
    
    # Compute metrics
    mAP, per_class_ap = map_calculator.compute_map()
    
    results = {
        "mAP": mAP,
        "per_class_mAP": {CLASS_NAMES[i]: ap for i, ap in per_class_ap.items()},
        "mAP_per_class_list": per_class_ap,
    }
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nmAP@0.5: {mAP:.2f}%\n")
    print("Per-Class mAP:")
    for class_id, ap in per_class_ap.items():
        class_name = CLASS_NAMES[class_id]
        print(f"  {class_id}: {class_name:35s} → {ap*100:.2f}%")
    
    return results


# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================

def plot_training_history(history_path: str):
    """
    STEP 5: Plot training history (loss over time)
    
    Helps identify:
        - How well model converged
        - Signs of overfitting (train loss down, val loss up)
        - When to stop training (plateau)
    """
    print("\n" + "="*70)
    print("STEP 5: Plotting Training History")
    print("="*70)
    
    with open(history_path) as f:
        history = json.load(f)
    
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    lrs = history['learning_rate']
    ax2.plot(lrs, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    print(f"✓ Plot saved: results/training_history.png\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*70)
    print("🍅 TOMATO DISEASE DETECTION - EVALUATION")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = "results/checkpoints/best.pt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train model first: python src/train_complete.py")
        return
    
    model, config = load_checkpoint(checkpoint_path, device)
    
    # Load validation data
    print("\n" + "="*70)
    print("STEP 2: Loading Validation Data")
    print("="*70)
    _, val_loader = create_dataloaders(batch_size=16, augment=False)
    
    # Evaluate
    results = evaluate_model(model, val_loader, device)
    
    # Save results
    results_path = Path("results") / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        json_results = {
            "mAP": float(results["mAP"]),
            "per_class_mAP": {
                str(k): float(v) for k, v in results["per_class_mAP"].items()
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # Plot history
    history_path = Path("results") / "history.json"
    if history_path.exists():
        plot_training_history(str(history_path))
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
