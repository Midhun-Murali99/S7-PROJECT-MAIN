import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.resnet50_classifier import TomatoResNet50

DATASET_PATH = PROJECT_ROOT / "dataset" / "tomato_yolo_dataset"
IMAGE_TRAIN = DATASET_PATH / "images" / "train"
IMAGE_VAL = DATASET_PATH / "images" / "val"
LABEL_TRAIN = DATASET_PATH / "labels" / "train"
LABEL_VAL = DATASET_PATH / "labels" / "val"

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
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_image_files(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_yolo_label(label_path: Path):
    if not label_path.exists():
        return []
    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) != 5:
                continue
            try:
                class_id = int(float(vals[0]))
                _, _, _, _ = map(float, vals[1:])
                x_center, y_center, width, height = map(float, vals[1:])
                annotations.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                })
            except ValueError:
                continue
    return annotations


class TomatoClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, transform=None):
        self.transform = transform
        self.samples = []

        for image_path in get_image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            labels = read_yolo_label(label_path)
            if labels:
                first_class = int(labels[0]["class_id"])
                self.samples.append((image_path, first_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

transform_eval = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def make_loaders(batch_size=32, num_workers=4):
    train_ds = TomatoClassificationDataset(IMAGE_TRAIN, LABEL_TRAIN, transform=transform_train)
    val_ds = TomatoClassificationDataset(IMAGE_VAL, LABEL_VAL, transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TomatoResNet50(num_classes=NUM_CLASSES, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    train_loader, val_loader = make_loaders(batch_size=32)
    results = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step()

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, results_dir / "resnet50_best.pt")

    with open(results_dir / "resnet50_history.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    print("Starting ResNet-50 tomato disease training...")
    train()
