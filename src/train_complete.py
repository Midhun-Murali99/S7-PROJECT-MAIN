"""
=============================================================================
TRAINING SCRIPT - YOLOv8 Object Detection
=============================================================================

This script trains a YOLOv8-style object detector on tomato disease data.

Training Pipeline:
    1. Load data via DataLoaders
    2. Create model and optimizer
    3. For each epoch:
        - Train on all training batches
        - Validate on validation set
        - Compute and log losses
        - Save best model
        - Check early stopping
    4. Evaluate final model

Key Concepts:
    - Epoch: One pass through entire training dataset
    - Batch: Subset of training data (size = batch_size)
    - Loss: Measure of prediction error
    - Gradient: Direction to update weights to reduce loss
    - Learning rate: Step size for weight updates
    - Validation: Check performance on unseen data
    - Early stopping: Stop if validation loss doesn't improve
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

# Import custom modules
from preprocessing.preprocess_complete import create_dataloaders, CLASS_NAMES, NUM_CLASSES
from models.yolov8_detector import YOLOv8ObjectDetector, YOLOLoss


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model": {
        "num_classes": NUM_CLASSES,
        "base_channels": 64,
    },
    "training": {
        "num_epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "warmup_epochs": 3,  # Gradually increase LR
    },
    "optimizer": {
        "name": "SGD",  # Stochastic Gradient Descent
        "momentum": 0.937,
        "weight_decay": 0.0005,
    },
    "scheduler": {
        "name": "cosine",  # Cosine annealing: decay LR smoothly
        "T_max": 97,  # T_max = total_epochs - warmup
    },
    "logging": {
        "log_interval": 10,  # Log every N batches
        "save_interval": 5,  # Save checkpoint every N epochs
    }
}


# ============================================================================
# STEP 1: SETUP DEVICE
# ============================================================================

def setup_device():
    """
    STEP 1: Detect and setup GPU/CPU
    
    Why GPU?
        - Modern neural networks have millions of parameters
        - Training on CPU: weeks to months
        - Training on GPU: hours to days
        - GPUs great at parallel matrix operations
    
    Returns:
        device: torch.device (cuda if available, else cpu)
    """
    print("\n" + "="*70)
    print("STEP 1: Setting up Device")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU not available - using CPU (training will be slow)")
    
    print(f"✓ Using device: {device}\n")
    return device


# ============================================================================
# STEP 2: CREATE MODEL
# ============================================================================

def create_model_and_optimizer(device):
    """
    STEP 2: Initialize model and optimizer
    
    What is an optimizer?
        - Algorithm to update weights based on gradients
        - Goal: minimize loss function
        - Common optimizers: SGD, Adam, AdamW
    
    SGD (Stochastic Gradient Descent):
        - Classic optimization algorithm
        - Update: w = w - lr * gradient
        - Momentum: m = β*m + gradient
        - Update: w = w - lr * m
        - Why effective: smooth updates, escape local minima
    
    Returns:
        model: YOLOv8ObjectDetector
        optimizer: torch.optim.SGD
        scheduler: Learning rate scheduler
    """
    print("\n" + "="*70)
    print("STEP 2: Creating Model and Optimizer")
    print("="*70)
    
    # Create model
    model = YOLOv8ObjectDetector(
        num_classes=CONFIG["model"]["num_classes"],
        base_channels=CONFIG["model"]["base_channels"]
    )
    model = model.to(device)
    
    print(f"✓ Model created with {model.count_parameters():,} parameters")
    print(f"✓ Model architecture layers:")
    print(f"  - Backbone (CSPDarknet)")
    print(f"  - Neck (Feature Pyramid)")
    print(f"  - Head (Detection Head)")
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        momentum=CONFIG["optimizer"]["momentum"],
        weight_decay=CONFIG["optimizer"]["weight_decay"]
    )
    
    print(f"\n✓ Optimizer: SGD")
    print(f"  - Learning rate: {CONFIG['training']['learning_rate']}")
    print(f"  - Momentum: {CONFIG['optimizer']['momentum']}")
    print(f"  - Weight decay: {CONFIG['optimizer']['weight_decay']}")
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["scheduler"]["T_max"],
        eta_min=0.0001
    )
    
    print(f"\n✓ Scheduler: Cosine Annealing")
    print(f"  - Gradually decreases LR during training")
    print(f"  - Helps escape local minima\n")
    
    return model, optimizer, scheduler


# ============================================================================
# STEP 3: TRAINING LOOP
# ============================================================================

class Trainer:
    """
    Trainer class to encapsulate training logic
    
    Responsibilities:
        - Load batches of data
        - Forward pass through model
        - Compute loss
        - Backward pass (compute gradients)
        - Update weights via optimizer
        - Track metrics
        - Save checkpoints
    """
    
    def __init__(self, model, optimizer, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Create loss function
        self.loss_fn = YOLOLoss(num_classes=config["model"]["num_classes"])
        self.loss_fn = self.loss_fn.to(device)
        
        # Tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20  # Early stopping patience
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Checkpoints directory
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
    
    def train_epoch(self, train_loader, epoch):
        """
        STEP 3a: Train for one epoch
        
        For each batch:
            1. Forward pass: x → model → predictions
            2. Compute loss: predictions vs ground truth
            3. Backward pass: compute gradients (chain rule)
            4. Optimization step: update weights with gradients
            5. Zero gradients: prepare for next batch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            avg_loss: Average loss over all batches
        """
        self.model.train()  # Set model to training mode
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Step 1: Move data to device
            images = images.to(self.device)
            
            # Step 2: Forward pass
            # images: (B, 3, 480, 480) → model → out3, out4, out5
            predictions = self.model(images)
            
            # Step 3: Compute loss
            # loss = measure of error
            loss = self.loss_fn(predictions, targets)
            
            # Step 4: Backward pass
            # Compute gradients using chain rule
            loss.backward()
            
            # Step 5: Optimization step
            # Update weights: w = w - lr * gradient
            self.optimizer.step()
            
            # Step 6: Zero gradients
            # Clear old gradients before next batch
            self.optimizer.zero_grad()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item():.4f})
            
            # Logging
            if (batch_idx + 1) % self.config["logging"]["log_interval"] == 0:
                avg_batch_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_batch_loss:.4f}, LR: {lr:.6f}")
        
        avg_loss = total_loss / num_batches
        
        # Step learning rate scheduler
        self.scheduler.step()
        
        return avg_loss
    
    @torch.no_grad()  # No gradient computation (faster)
    def validate(self, val_loader, epoch):
        """
        STEP 3b: Validate on validation set
        
        Why validate?
            - Check performance on unseen data
            - Detect overfitting (train loss low, val loss high)
            - Select best model based on validation loss
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()  # Set model to evaluation mode
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}")
        
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Forward pass (no gradient computation)
            predictions = self.model(images)
            
            # Compute loss
            loss = self.loss_fn(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item():.4f})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        STEP 3c: Save model checkpoint
        
        What to save:
            - Model weights
            - Optimizer state (momentum, etc.)
            - Scheduler state (current LR)
            - Epoch number
            - Training history
        
        Why save checkpoints?
            - Resume training if interrupted
            - Load best model for inference
            - Analyze training progress
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
    
    def train(self, train_loader, val_loader):
        """
        STEP 3: Complete training loop
        
        For each epoch:
            1. Train on all training data
            2. Validate on validation data
            3. Track metrics
            4. Save checkpoints
            5. Check early stopping
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        """
        print("\n" + "="*70)
        print("STEP 3: Starting Training")
        print("="*70 + "\n")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"✓ Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            print(f"✓ Val Loss: {val_loss:.4f}")
            
            # Track metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.config["logging"]["save_interval"] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*70)
        print("✓ Training Complete!")
        print("="*70)
        
        # Save history
        history_path = self.results_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved: {history_path}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🍅 TOMATO DISEASE DETECTION - TRAINING")
    print("="*70)
    
    # Setup
    device = setup_device()
    
    # Create model and optimizer
    model, optimizer, scheduler = create_model_and_optimizer(device)
    
    # Create dataloaders
    print("\n" + "="*70)
    print("STEP 2.1: Loading Data")
    print("="*70)
    train_loader, val_loader = create_dataloaders(
        batch_size=CONFIG["training"]["batch_size"],
        augment=True
    )
    
    # Create trainer
    trainer = Trainer(model, optimizer, scheduler, device, CONFIG)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n" + "="*70)
    print("✓ ALL STEPS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review training history: results/history.json")
    print("  2. Evaluate model: python src/evaluate.py")
    print("  3. Run inference: python src/inference.py --image <path>")


if __name__ == "__main__":
    main()
