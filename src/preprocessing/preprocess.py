from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Project root:
# S7-PROJECT-MAIN/
#
# This file:
# src/preprocessing/preprocessing.py
#
# Dataset:
# dataset/tomato_yolo_dataset/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "tomato_yolo_dataset"

IMAGE_TRAIN = DATASET_PATH / "images" / "train"
IMAGE_VAL = DATASET_PATH / "images" / "val"

LABEL_TRAIN = DATASET_PATH / "labels" / "train"
LABEL_VAL = DATASET_PATH / "labels" / "val"


# ============================================================
# CLASS INFORMATION
# ============================================================

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


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

# For now we use only resizing and normalization.
#
# We are NOT adding aggressive augmentation yet.
# Augmentation will be added after the dataset validation
# is confirmed to be correct.

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET STRUCTURE CHECK
# ============================================================

def check_dataset_structure():
    """
    Check whether the expected YOLO dataset structure exists.
    """

    print("\nChecking tomato dataset...")

    paths = {
        "Dataset path": DATASET_PATH,
        "Training images": IMAGE_TRAIN,
        "Validation images": IMAGE_VAL,
        "Training labels": LABEL_TRAIN,
        "Validation labels": LABEL_VAL,
    }

    all_exist = True

    for name, path in paths.items():
        exists = path.exists()

        if name == "Dataset path":
            print(f"{name}: {exists}")
        else:
            print(f"{name}: {exists}")

        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError(
            "\nDataset structure is incomplete.\n"
            f"Expected dataset at:\n{DATASET_PATH}"
        )

    return True


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(image_directory):
    """
    Return all supported image files from a directory.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_files = [
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    return sorted(image_files)


# ============================================================
# GET CORRESPONDING LABEL FILE
# ============================================================

def get_label_path(image_path, label_directory):
    """
    Given an image path, find its corresponding YOLO label file.

    Example:

    image:
    TMBS_image (1).jpg

    label:
    TMBS_image (1).txt
    """

    return label_directory / f"{image_path.stem}.txt"


# ============================================================
# READ YOLO LABEL
# ============================================================

def read_yolo_label(label_path):
    """
    Read a YOLO annotation file.

    YOLO format:

    class_id x_center y_center width height

    Example:

    0 0.523 0.421 0.634 0.721

    Returns:
        list of dictionaries containing class and bounding box
    """

    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            values = line.split()

            # YOLO annotation must contain exactly 5 values
            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path} "
                    f"at line {line_number}: {line}"
                )

            try:
                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError:
                raise ValueError(
                    f"Non-numeric YOLO annotation in "
                    f"{label_path} at line {line_number}"
                )

            # ------------------------------------------------
            # Validate class
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class ID {class_id} in {label_path}"
                )

            # ------------------------------------------------
            # Validate YOLO coordinates
            # ------------------------------------------------

            coordinates = [
                x_center,
                y_center,
                width,
                height
            ]

            for value in coordinates:

                if not 0.0 <= value <= 1.0:

                    raise ValueError(
                        f"Invalid YOLO coordinate {value} "
                        f"in {label_path}"
                    )

            # Width and height must be greater than zero
            if width <= 0 or height <= 0:

                raise ValueError(
                    f"Invalid bounding box size in {label_path}"
                )

            annotations.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    return annotations


# ============================================================
# VALIDATE ONE IMAGE
# ============================================================

def validate_image(image_path):
    """
    Check whether an image can be opened successfully.
    """

    try:

        with Image.open(image_path) as image:

            image.verify()

        return True

    except Exception as error:

        print(
            f"Corrupt/unreadable image: "
            f"{image_path}"
        )

        print(f"Reason: {error}")

        return False


# ============================================================
# VALIDATE DATASET SPLIT
# ============================================================

def validate_split(image_directory, label_directory, split_name):
    """
    Validate all images and labels in one dataset split.
    """

    print("\n" + "=" * 60)
    print(f"VALIDATING {split_name.upper()} DATA")
    print("=" * 60)

    images = get_image_files(image_directory)

    print(f"Images found: {len(images)}")

    missing_labels = []
    corrupt_images = []
    invalid_labels = []
    empty_labels = []

    class_counts = Counter()

    total_objects = 0

    # --------------------------------------------------------
    # Check every image
    # --------------------------------------------------------

    for image_path in images:

        # Check corresponding label
        label_path = get_label_path(
            image_path,
            label_directory
        )

        if not label_path.exists():

            missing_labels.append(image_path.name)

            continue

        # Check image integrity
        if not validate_image(image_path):

            corrupt_images.append(image_path.name)

            continue

        # Check YOLO label
        try:

            annotations = read_yolo_label(label_path)

        except Exception as error:

            invalid_labels.append(
                (label_path.name, str(error))
            )

            continue

        # Empty label file
        if len(annotations) == 0:

            empty_labels.append(label_path.name)

            continue

        # Count objects/classes
        total_objects += len(annotations)

        for annotation in annotations:

            class_id = annotation["class_id"]

            class_counts[class_id] += 1

    # --------------------------------------------------------
    # Print validation results
    # --------------------------------------------------------

    print("\nValidation results:")

    print(f"Missing labels: {len(missing_labels)}")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Invalid labels: {len(invalid_labels)}")
    print(f"Empty labels: {len(empty_labels)}")
    print(f"Total annotated objects: {total_objects}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id in range(NUM_CLASSES):

        count = class_counts[class_id]

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]} : "
            f"{count}"
        )

    # --------------------------------------------------------
    # Print problems
    # --------------------------------------------------------

    if missing_labels:

        print("\nMissing label examples:")

        for filename in missing_labels[:10]:
            print(f"  {filename}")

    if corrupt_images:

        print("\nCorrupt image examples:")

        for filename in corrupt_images[:10]:
            print(f"  {filename}")

    if invalid_labels:

        print("\nInvalid label examples:")

        for filename, error in invalid_labels[:10]:

            print(f"  {filename}")
            print(f"    {error}")

    if empty_labels:

        print("\nEmpty label examples:")

        for filename in empty_labels[:10]:
            print(f"  {filename}")

    return {
        "images": len(images),
        "missing_labels": missing_labels,
        "corrupt_images": corrupt_images,
        "invalid_labels": invalid_labels,
        "empty_labels": empty_labels,
        "class_counts": class_counts,
        "total_objects": total_objects,
    }


# ============================================================
# FULL DATASET VALIDATION
# ============================================================

def validate_dataset():
    """
    Validate both training and validation datasets.
    """

    check_dataset_structure()

    train_results = validate_split(
        IMAGE_TRAIN,
        LABEL_TRAIN,
        "train"
    )

    val_results = validate_split(
        IMAGE_VAL,
        LABEL_VAL,
        "validation"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"Training images: "
        f"{train_results['images']}"
    )

    print(
        f"Validation images: "
        f"{val_results['images']}"
    )

    print(
        f"Training missing labels: "
        f"{len(train_results['missing_labels'])}"
    )

    print(
        f"Validation missing labels: "
        f"{len(val_results['missing_labels'])}"
    )

    print(
        f"Training corrupt images: "
        f"{len(train_results['corrupt_images'])}"
    )

    print(
        f"Validation corrupt images: "
        f"{len(val_results['corrupt_images'])}"
    )

    print(
        f"Training invalid labels: "
        f"{len(train_results['invalid_labels'])}"
    )

    print(
        f"Validation invalid labels: "
        f"{len(val_results['invalid_labels'])}"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    problems = (
        len(train_results["missing_labels"])
        + len(val_results["missing_labels"])
        + len(train_results["corrupt_images"])
        + len(val_results["corrupt_images"])
        + len(train_results["invalid_labels"])
        + len(val_results["invalid_labels"])
    )

    if problems == 0:

        print("\nDataset validation completed successfully.")
        print("No missing labels, corrupt images, or invalid labels found.")

    else:

        print(
            f"\nDataset validation completed with "
            f"{problems} problem(s)."
        )

    return train_results, val_results


# ============================================================
# PYTORCH DATASET
# ============================================================

class TomatoDataset(Dataset):
    """
    PyTorch Dataset for the tomato YOLO dataset.

    Each item returns:

        image
        label
        image_path
        annotations
    """

    def __init__(
        self,
        image_directory,
        label_directory,
        transform=None
    ):

        self.image_directory = Path(image_directory)
        self.label_directory = Path(label_directory)

        self.transform = transform

        self.image_files = get_image_files(
            self.image_directory
        )

        # ----------------------------------------------------
        # Keep only images that have labels
        # ----------------------------------------------------

        self.samples = []

        for image_path in self.image_files:

            label_path = get_label_path(
                image_path,
                self.label_directory
            )

            if label_path.exists():

                annotations = read_yolo_label(
                    label_path
                )

                if len(annotations) > 0:

                    self.samples.append(
                        (image_path, label_path)
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label_path = self.samples[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Read YOLO annotations
        # ----------------------------------------------------

        annotations = read_yolo_label(
            label_path
        )

        # ----------------------------------------------------
        # For the current classification pipeline:
        #
        # use the first object's class as the image label.
        #
        # We will revisit this when handling multiple
        # objects and temporal sequences.
        # ----------------------------------------------------

        label = annotations[0]["class_id"]

        # ----------------------------------------------------
        # Apply image transformation
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "image_path": str(image_path),
            "annotations": annotations
        }


# ============================================================
# DATASET CREATION FUNCTIONS
# ============================================================

def create_train_dataset():

    return TomatoDataset(
        IMAGE_TRAIN,
        LABEL_TRAIN,
        transform=IMAGE_TRANSFORM
    )


def create_validation_dataset():

    return TomatoDataset(
        IMAGE_VAL,
        LABEL_VAL,
        transform=IMAGE_TRANSFORM
    )


# ============================================================
# SAMPLE CHECK
# ============================================================

def check_sample(dataset):
    """
    Load one sample and print its information.
    """

    if len(dataset) == 0:

        print("Dataset contains no valid samples.")

        return

    sample = dataset[0]

    print("\n" + "=" * 60)
    print("SAMPLE CHECK")
    print("=" * 60)

    print(
        f"Image shape: "
        f"{sample['image'].shape}"
    )

    print(
        f"Label: "
        f"{sample['label'].item()}"
    )

    print(
        f"Class: "
        f"{CLASS_NAMES[sample['label'].item()]}"
    )

    print(
        f"Image: "
        f"{sample['image_path']}"
    )

    print(
        f"Objects: "
        f"{len(sample['annotations'])}"
    )

    print("\nAnnotations:")

    for annotation in sample["annotations"]:

        print(
            f"  Class {annotation['class_id']} "
            f"→ {CLASS_NAMES[annotation['class_id']]}"
        )

        print(
            f"    Box: "
            f"x={annotation['x_center']:.3f}, "
            f"y={annotation['y_center']:.3f}, "
            f"w={annotation['width']:.3f}, "
            f"h={annotation['height']:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Validate complete dataset
    # --------------------------------------------------------

    train_results, val_results = validate_dataset()

    # --------------------------------------------------------
    # 2. Create PyTorch datasets
    # --------------------------------------------------------

    train_dataset = create_train_dataset()

    val_dataset = create_validation_dataset()

    print("\n" + "=" * 60)
    print("PYTORCH DATASET")
    print("=" * 60)

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Number of classes: "
        f"{NUM_CLASSES}"
    )

    print("\nClasses:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_id}: {class_name}"
        )

    # --------------------------------------------------------
    # 3. Check one training sample
    # --------------------------------------------------------

    check_sample(train_dataset)