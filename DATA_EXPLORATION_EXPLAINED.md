# Data Exploration Notebook - Step-by-Step Explanation

## Overview
This Jupyter notebook performs **Exploratory Data Analysis (EDA)** on the tomato disease detection dataset.

EDA helps us understand:
- How many images and objects we have
- Class distribution (are some diseases more common?)
- Image sizes and quality
- Bounding box sizes (are objects small or large?)
- Data issues to fix before training

---

## Cell 1: Imports
```python
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import random
```

**What's happening:**
- Import libraries for file handling, visualization, and data structures

**Key imports:**
- `Path`: Modern file path handling (better than strings)
- `matplotlib`: Plotting library for visualization
- `PIL.Image`: Loading and manipulating images
- `Counter`: Count occurrences of items (useful for class distribution)
- `random`: Random sampling for visualization

---

## Cell 2: Setup Paths
```python
DATASET_PATH = Path("../dataset/tomato_yolo_dataset")

IMAGE_TRAIN = DATASET_PATH / "images" / "train"
IMAGE_VAL = DATASET_PATH / "images" / "val"
LABEL_TRAIN = DATASET_PATH / "labels" / "train"
LABEL_VAL = DATASET_PATH / "labels" / "val"

print("Dataset exists:", DATASET_PATH.exists())
print("Train images:", IMAGE_TRAIN.exists())
print("Validation images:", IMAGE_VAL.exists())
print("Train labels:", LABEL_TRAIN.exists())
print("Validation labels:", LABEL_VAL.exists())
```

**What's happening:**
- Define paths to dataset folders
- Verify all paths exist (sanity check)

**YOLO Directory Structure:**
```
tomato_yolo_dataset/
├── images/
│   ├── train/    ← Training images (used to teach model)
│   └── val/      ← Validation images (used to check performance)
└── labels/
    ├── train/    ← Bounding box labels for training images
    └── val/      ← Bounding box labels for validation images
```

**Output check:**
- All should print `True`
- If any are `False`, dataset is incomplete

---

## Cell 3: Count Images
```python
train_images = list(IMAGE_TRAIN.glob("*"))
val_images = list(IMAGE_VAL.glob("*"))

print("Training images:", len(train_images))
print("Validation images:", len(val_images))
```

**What's happening:**
- Count total images in train and val folders
- Glob pattern `*` matches all files

**Typical output:**
```
Training images: 800
Validation images: 200
```

**Why count?**
- More data = better training
- 80/20 train/val split is standard
- With ~1000 images, we have moderate dataset size

---

## Cell 4: Analyze Class Distribution
```python
class_ids = Counter()

for label_file in LABEL_TRAIN.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 5:
                class_id = int(values[0])
                class_ids[class_id] += 1

print(class_ids)
```

**What's happening:**
1. Loop through all YOLO label files
2. For each line (each bounding box)
3. Extract class_id (first value)
4. Count occurrences → see which diseases are most common

**YOLO Label Format:**
```
class_id x_center y_center width height

Example:
0 0.523 0.421 0.634 0.721  ← Disease (class 0) at center
```

**Typical output:**
```
Counter({9: 450, 0: 120, 1: 150, 2: 100, ...})
```
"Class 9 (Healthy) appears 450 times, class 0 (Bacterial spot) 120 times, etc."

**Why analyze?**
- Imbalanced data (some diseases rare)
- Need weighted loss during training (give more weight to rare classes)
- Can use data augmentation to oversample rare classes

---

## Cell 5: Display Class Names
```python
class_names = {
    0: "Bacterial spot",
    1: "Early blight",
    2: "Late blight",
    3: "Leaf Mold",
    4: "Septoria leaf spot",
    5: "Spider mites",
    6: "Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato mosaic virus",
    9: "Healthy"
}

for class_id, count in sorted(class_ids.items()):
    print(class_id, "→", class_names.get(class_id, "Unknown"), ":", count)
```

**What's happening:**
- Print human-readable class names with counts
- Sort by class_id for clarity

**Output example:**
```
0 → Bacterial spot → 120
1 → Early blight → 150
2 → Late blight → 100
...
9 → Healthy → 450
```

**Why display?**
- Understand dataset composition
- See which diseases are well-represented
- Plan augmentation strategy

---

## Cell 6: Visualize Sample Images
```python
import matplotlib.pyplot as plt
from PIL import Image
import random

# Pick 6 random training images
sample_images = random.sample(train_images, 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, image_path in zip(axes.ravel(), sample_images):
    image = Image.open(image_path)
    ax.imshow(image)
    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

**What's happening:**
1. Randomly select 6 training images
2. Create 2×3 grid of subplots
3. Display each image
4. Set title to filename

**Why visualize?**
- See raw images (do they look good?)
- Check for quality issues (blur, darkness, noise)
- Verify correct disease samples
- Assess image variety (different angles, lighting)

---

## Cell 7: Visualize Bounding Boxes
```python
from PIL import ImageDraw

# Take one random image
image_path = random.choice(train_images)
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Corresponding label file
label_path = LABEL_TRAIN / (image_path.stem + ".txt")

# Image dimensions
img_width, img_height = image.size

# Read YOLO labels
with open(label_path, "r") as f:
    for line in f:
        values = line.strip().split()
        if len(values) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, values)

        # Convert YOLO coordinates (normalized) to pixel coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate corner coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Label the box
        label = class_names[int(class_id)]
        draw.text((x1, y1), label, fill="red")

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis("off")
plt.show()
```

**What's happening:**
1. Load a random image and its label file
2. For each bounding box in the image:
   - Read YOLO format (normalized: 0-1)
   - Convert to pixel coordinates
   - Draw red rectangle
   - Add disease name label

**YOLO Coordinate Conversion:**
```
YOLO (normalized):  [0.5, 0.5, 0.6, 0.7]
                    x_center, y_center, width, height (as fraction of image)

To pixel coordinates:
x_center_px = 0.5 × 1920 = 960
y_center_px = 0.5 × 1080 = 540
width_px = 0.6 × 1920 = 1152
height_px = 0.7 × 1080 = 756

Top-left corner:
x1 = 960 - 1152/2 = 384
y1 = 540 - 756/2 = 162
```

**Why visualize bounding boxes?**
- Verify annotations are correct
- Check box quality (tight around objects)
- Detect mislabeled images
- Understand disease detection task

---

## Cell 8: Count Objects per Image
```python
object_counts = []

for label_file in LABEL_TRAIN.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    object_counts.append(len(lines))

print("Total labeled images:", len(object_counts))
print("Images with 1 object:", object_counts.count(1))
print("Images with 2 objects:", object_counts.count(2))
print("Images with 3 objects:", object_counts.count(3))
print("Images with more than 3 objects:", sum(1 for x in object_counts if x > 3))
```

**What's happening:**
- For each image, count number of bounding boxes (objects)
- Analyze distribution

**Typical output:**
```
Total labeled images: 800
Images with 1 object: 450   (56%)
Images with 2 objects: 250   (31%)
Images with 3 objects: 80    (10%)
Images with more than 3 objects: 20 (2%)
```

**Why analyze?**
- Most images have single disease (classification-like)
- Some have multiple diseases (complex scenes)
- Affects model training strategy

---

## Cell 9: Image Size Distribution
```python
from collections import Counter

image_sizes = Counter()

for image_path in train_images:
    try:
        with Image.open(image_path) as img:
            image_sizes[img.size] += 1
    except:
        pass

print("Number of different image sizes:", len(image_sizes))
print("\nMost common image sizes:")
for size, count in image_sizes.most_common(10):
    print(size, "→", count)
```

**What's happening:**
- Load each image and record its dimensions
- Count how many images have each size

**Typical output:**
```
Number of different image sizes: 2
Most common image sizes:
(1920, 1080) → 650   (Most images)
(1280, 720) → 150    (Some images)
```

**Why analyze?**
- Images have different sizes → need to resize before training
- Standard practice: resize all to 480×480
- Resize: loses information but standardizes model input

---

## Cell 10: Bounding Box Statistics
```python
box_areas = []

for label_file in LABEL_TRAIN.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 5:
                continue

            _, _, _, width, height = map(float, values)
            # YOLO width and height are fractions of the image
            area = width * height
            box_areas.append(area)

print("Number of bounding boxes:", len(box_areas))
print("Average box area:", sum(box_areas) / len(box_areas))
print("Smallest box area:", min(box_areas))
print("Largest box area:", max(box_areas))
```

**What's happening:**
- For each bounding box, calculate area (width × height)
- Compute statistics

**Box Area = width × height (both normalized 0-1)**

**Typical output:**
```
Number of bounding boxes: 950
Average box area: 0.08   (8% of image)
Smallest box area: 0.01  (small diseased spots)
Largest box area: 0.45   (large affected areas)
```

**Why analyze?**
- Objects range from small spots to large areas
- Small objects harder to detect (need high resolution)
- Multi-scale detection important for this dataset

---

## Cell 11: Test Classification Conversion
```python
from pathlib import Path
import shutil

dataset_path = Path("../dataset/tomato_yolo_dataset")
train_images = dataset_path / "images" / "train"
train_labels = dataset_path / "labels" / "train"

class_names = {
    0: "Bacterial_spot",
    1: "Early_blight",
    # ... all 10 classes ...
    9: "Healthy"
}

label_files = list(train_labels.glob("*.txt"))[:20]

print("Testing conversion on", len(label_files), "images")

for label_file in label_files:
    # Read YOLO annotation
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Only use images with exactly one object
    if len(lines) != 1:
        continue

    # Get class ID
    class_id = int(lines[0].split()[0])
    class_name = class_names[class_id]

    # Find corresponding image
    image_file = train_images / (label_file.stem + ".jpg")

    if not image_file.exists():
        continue

    print(f"{image_file.name}  →  {class_name}")
```

**What's happening:**
1. For images with exactly ONE object
2. Extract the disease class
3. Print mapping: "TMBS_image (1).jpg → Bacterial spot"

**Why do this?**
- **Hybrid approach**: Object detection OR classification
- Single-object images → can use as classification
- Multi-object images → need object detection
- Different models for different scenarios

---

## Summary of Insights

After running all cells, you should understand:

| Metric | Typical Value | Implication |
|--------|---------------|------------|
| Train images | ~800 | Moderate dataset, might need augmentation |
| Val images | ~200 | Good for validation |
| Total objects | ~950 | Enough for training |
| Class imbalance | 9 appears 450x, others 50-150x | Use weighted loss |
| Image sizes | 1920×1080, 1280×720 | Resize to 480×480 |
| Objects/image | 1-3 usually | Good for detection |
| Box area | 1%-45% | Mix of small and large |

---

## Next Steps

1. **Data Augmentation**: Add random flips, rotations, brightness changes
2. **Data Cleaning**: Remove mislabeled or corrupted images
3. **Class Balancing**: Use weighted loss or oversampling
4. **Model Selection**: YOLOv8 works well for this dataset
5. **Training**: Use the prepared preprocessing and training scripts
