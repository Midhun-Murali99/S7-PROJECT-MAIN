# Quick Start Guide - Tomato Disease Classification

## 1. What this project does

This project uses a convolutional neural network (CNN) to classify a tomato
leaf image into one of 10 classes:

1. Bacterial spot
2. Early blight
3. Late blight
4. Leaf Mold
5. Septoria leaf spot
6. Spider mites
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato mosaic virus
10. Healthy

The current primary workflow is **image classification with a ResNet-50 CNN**.
The model receives one complete image and returns one class prediction. It does
not currently perform object detection, draw predicted bounding boxes, or
calculate detection mAP.

The dataset keeps labels in YOLO format because the original annotations
contain bounding boxes. In the current classification pipeline, the first
annotation in each non-empty label file supplies the image class. The bounding
box coordinates are validated and retained as metadata, but they are not
cropped or used as detection targets during CNN training.

## 2. Repository layout

```text
S7 PROJECT MAIN/
|-- dataset/
|   `-- tomato_yolo_dataset/
|       |-- images/
|       |   |-- train/
|       |   `-- val/
|       `-- labels/
|           |-- train/
|           `-- val/
|-- src/
|   |-- preprocessing/
|   |   `-- preprocess.py
|   |-- classification/
|   |   `-- resnet50_classifier.py
|   |-- infer.py
|   `-- classification/train.py
|-- notebooks/
|   `-- data_exploration.ipynb
|-- results/
|-- requirements.txt
`-- QUICK_START_GUIDE.md
```

## 3. Environment setup

Run these commands from the repository root in PowerShell:

```powershell
cd "S:\S7 PROJECT MAIN"

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with the virtual
environment's interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependencies are:

- PyTorch and torchvision for the CNN, tensors, training, and inference
- Pillow for image loading
- NumPy for numerical utilities
- pandas and Matplotlib for notebook analysis and visualizations

CUDA is optional. PyTorch uses the GPU when CUDA is available; otherwise the
scripts fall back to the CPU.

## 4. Dataset format

The expected directory structure is:

```text
dataset/tomato_yolo_dataset/
|-- images/
|   |-- train/*.jpg
|   `-- val/*.jpg
`-- labels/
    |-- train/*.txt
    `-- val/*.txt
```

Every image should have a label file with the same filename stem:

```text
images/train/TMBS_image (1).jpg
labels/train/TMBS_image (1).txt
```

Each label line follows the YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the range 0.0 to 1.0. For example:

```text
0 0.523 0.421 0.634 0.721
```

The validation code checks that:

- the dataset directories exist;
- supported image files can be found;
- every image has a matching label file;
- images can be opened successfully;
- each label line has exactly five values;
- class IDs are between 0 and 9;
- coordinates are numeric and within the normalized range; and
- bounding-box width and height are greater than zero.

Images without a valid, non-empty label are skipped by the classification
dataset. If a label file contains multiple objects, the first object's
`class_id` is used as the image label. This is an intentional classification
simplification, not multi-object detection.

## 5. Explore and validate the data

Run the exploration notebook before training:

```powershell
jupyter notebook notebooks\data_exploration.ipynb
```

The notebook checks the dataset structure, image/label matching, class
balance, invalid or empty labels, image dimensions, train/validation
distribution, and representative YOLO bounding boxes. It is an inspection
tool only and does not train the CNN.

Run the repository's dataset validator:

```powershell
python src\preprocessing\preprocess.py
```

The validator prints image counts, missing labels, corrupt images, invalid
labels, empty labels, total annotated objects, and per-class counts. Fix
validation errors before training.

The normal single-image tensor shape is `(3, 224, 224)`. A batch passed to the
CNN has shape `(batch_size, 3, 224, 224)`, and the classifier output has shape
`(batch_size, 10)`.

## 6. CNN preprocessing

All classification models use RGB images resized to **224 x 224** pixels.
Evaluation preprocessing is:

1. Open the image with Pillow and convert it to RGB.
2. Resize it to `(224, 224)`.
3. Convert it to a PyTorch tensor.
4. Normalize channels with ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The ResNet-50 training path additionally applies training augmentation:

- random horizontal flip (50% probability);
- random rotation up to 15 degrees; and
- random brightness and contrast changes.

Augmentation is applied only to training images. Validation images use the
deterministic resize and normalization pipeline.

## 7. Recommended training: ResNet-50 CNN

The recommended end-to-end training entry point is:

```powershell
python src\classification\train.py
```

The script:

1. reads the training and validation image/label directories;
2. creates the classification datasets and data loaders;
3. builds a ResNet-50 CNN with its final layer changed to 10 outputs;
4. uses cross-entropy loss;
5. trains with AdamW (`learning_rate=1e-4`, `weight_decay=1e-4`);
6. reduces the learning rate with cosine annealing; and
7. records loss and validation accuracy for each epoch.

The default configuration is 8 epochs and a batch size of 32. The best model
is selected using the lowest validation loss and saved as:

```text
results/resnet50_best.pt
```

Training history is saved as:

```text
results/resnet50_history.json
```

The script selects CUDA automatically when available. To change the default
batch size, epoch count, or optimizer settings, edit the configuration in
`src\classification\train.py`.

### How the ResNet-50 CNN works

```text
Input RGB image (3 x 224 x 224)
              |
              v
       ResNet-50 convolutional backbone
       - convolution filters learn visual features
       - residual blocks improve deep-network training
       - progressively captures edges, textures, and disease patterns
              |
              v
       Global feature representation
              |
              v
       Fully connected layer: 2048 features -> 10 classes
              |
              v
       Class logits -> softmax probabilities
```

During training, cross-entropy compares the logits with the class label.
Backpropagation computes gradients, and AdamW updates the CNN weights.
Validation accuracy and validation loss measure generalization to images not
used for weight updates.

## 8. ResNet-50 inference

Run inference with the checkpoint produced by the ResNet-50 training script:

```powershell
python src\infer.py `
  --image "dataset\tomato_yolo_dataset\images\val\example.jpg" `
  --model results\resnet50_best.pt `
  --device cpu `
  --topk 3
```

The output contains the top-k class names, class IDs, and probabilities.
Probabilities are calculated by applying softmax to the 10 CNN logits.
The inference script and training script both use the same `TomatoResNet50`
architecture, so their checkpoint formats are compatible.

## 10. Evaluation and metrics

The active classifier is evaluated with:

- validation loss, where lower is better;
- top-1 accuracy, the percentage of images whose highest-probability class is
  correct; and
- optionally top-k accuracy when inspecting inference results.

Classification accuracy is the appropriate primary metric for the current
pipeline. Object-detection metrics such as IoU, mAP@0.5, and non-maximum
suppression belong to the legacy YOLO-style detector files and do not describe
the ResNet classifier's output.

## 11. Common troubleshooting

### `ModuleNotFoundError`

Run commands from the repository root and activate the virtual environment:

```powershell
cd "S:\S7 PROJECT MAIN"
.venv\Scripts\Activate.ps1
```

### Dataset structure is incomplete

Confirm that all four directories exist:

```text
dataset\tomato_yolo_dataset\images\train
dataset\tomato_yolo_dataset\images\val
dataset\tomato_yolo_dataset\labels\train
dataset\tomato_yolo_dataset\labels\val
```

### CUDA out of memory

Reduce the batch size in the training script, for example from 32 to 16 or 8.
The model can be trained on CPU, although it will be slower.

### Training accuracy is high but validation accuracy is low

- check for duplicate or near-duplicate images across splits;
- inspect class balance with the validator;
- confirm that image and label stems match;
- review labels with multiple objects, since only the first class is used; and
- collect more varied examples or adjust augmentation.

### Checkpoint loading fails

Use the ResNet-50 checkpoint produced by the active training script. Model
weights from another architecture cannot be loaded directly.

## 12. Important project limitations

- The current primary model classifies one complete image; it does not localize
  multiple diseases within one image.
- YOLO bounding boxes are not cropped or passed to a detector during CNN
  training.
- When an image has multiple annotations, only the first annotation's class is
  used.
- Reported performance must be measured on this dataset and should not be
  replaced with generic or assumed accuracy/mAP values.

## 13. Useful files

| File | Purpose |
|---|---|
| `src\preprocessing\preprocess.py` | Dataset paths, class names, label parsing, validation, and transforms |
| `src\classification\train.py` | ResNet-50 CNN training script |
| `src\classification\resnet50_classifier.py` | ResNet-50 model definition |
| `src\infer.py` | ResNet-50 checkpoint inference |
| `notebooks\data_exploration.ipynb` | Dataset exploration and visualization |

## 14. Quick checklist

- [ ] Create and activate the virtual environment.
- [ ] Install `requirements.txt`.
- [ ] Confirm the four dataset directories exist.
- [ ] Run `python src\preprocessing\preprocess.py`.
- [ ] Train with `python src\classification\train.py`.
- [ ] Run inference with `python src\infer.py`.
- [ ] Evaluate on the validation split and record measured metrics.
