Repository-specific Copilot instructions

This file is intended to make future Copilot sessions (and other AI assistants) productive quickly when working with this repository.

1) Install / environment

- Install runtime dependencies: from the repository root run:
  - python -m pip install -r requirements.txt

2) Build / test / lint commands

- There are no formal build / test / lint runners in this repository (no pytest/tox/flake8/Makefile detected).
- Useful quick commands (run from repo root):
  - Validate dataset (full check):
    - python src\preprocessing\preprocess.py
  - Inspect dataset sizes / create simple PyTorch datasets (single-sample check):
    - python src\dataset\sequence_dataset.py
  - Install dependencies (one-off):
    - python -m pip install -r requirements.txt

- Notes about tests: there are no unit tests detected in the repo. If tests are added later, place them under a tests/ or src/<package>/tests/ directory and use pytest -k <pattern> to run a single test by name/pattern.

3) High-level architecture (big picture)

- Purpose: image-based disease classification for tomato images using YOLO-style annotations (bounding boxes in normalized coordinates) and a classification pipeline that currently treats each image as a single labeled example.

- Core components:
  - dataset/tomato_yolo_dataset/: dataset root (expected structure)
    - images/train, images/val
    - labels/train, labels/val
  - src/preprocessing/preprocess.py
    - Dataset validation utilities, YOLO label parsing, a PyTorch Dataset (TomatoDataset) and helpers to create train/val datasets and run a sample check. Running this script performs a dataset validation and prints summaries.
    - Important constants: IMAGE_SIZE = 224, IMAGE_TRANSFORM (Resize -> ToTensor -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])). CLASS_NAMES maps class_id → human-readable name and NUM_CLASSES gives the class count.
  - src/dataset/sequence_dataset.py
    - Lightweight Dataset wrapper that creates samples for single images or short sequences (sequence_length parameter). It reads YOLO-format labels and returns samples of the shape the model pipeline expects (supports sequence_length > 1 by stacking identical frames; designed to keep API stable for future spatiotemporal training).
  - src/models/
    - spatial_encoder.py and spatiotemporal_transformer.py are present as model entry points (currently empty/skeletons). Expected flow: spatial encoder encodes per-frame features, a spatiotemporal transformer consumes sequences of frame-level features.
  - Training and evaluation entrypoints:
    - src/train.py and src/evaluate.py are present but currently have no implementation (placeholders). The preprocess and dataset modules are the primary runnable code today.
  - notebooks/
    - Jupyter notebooks (data_exploration.ipynb, preprocessing.ipynb) contain exploratory data analysis and are useful references for data shape and label handling.

4) Key repository conventions / non-obvious patterns

- YOLO annotation format is required (per-line: class_id x_center y_center width height) with coordinates normalized to [0.0, 1.0]. The code validates that lines have 5 numeric values, class_id is known, and coordinates are within [0,1]. Width and height must be > 0.

- Image ↔ label pairing:
  - Labels must be named to match the image stem (example: TMBS_image (1).jpg → TMBS_image (1).txt). The code constructs label paths using label_directory / image_path.stem + '.txt'.

- Label handling for classification:
  - When there are multiple objects annotated in one image, the current classification pipeline uses the first object (annotations[0]) as the image label. This is deliberate and is called out in the dataset code; any multi-object logic must be added explicitly.

- Dataset filtering:
  - The PyTorch Dataset classes (TomatoDataset and SequenceDataset) only include samples which have a non-empty .txt label file and valid annotations. Images without labels or with empty labels are skipped.

- Transform & normalization:
  - IMAGE_SIZE is 224 and the normalization statistics follow ImageNet defaults (mean/std). These are applied by default via IMAGE_TRANSFORM in preprocess.py.

- Project root detection:
  - Many modules determine PROJECT_ROOT using: Path(__file__).resolve().parents[2]. Scripts assume running from the repository root or that Python's import path allows importing src modules. If running scripts from a different CWD, ensure the repo root is on PYTHONPATH or run them from the repository root.

- Dataset validation CLI behavior:
  - preprocess.py's main checks the dataset structure and will raise FileNotFoundError if the expected tomato_yolo_dataset structure is missing. The script prints counts of missing labels, corrupt images, invalid label lines, class distribution, and example filenames for issues.

5) Where to look next (quick pointers)

- Data validation & dataset utilities: src\preprocessing\preprocess.py
- Dataset wrapper & sequence support: src\dataset\sequence_dataset.py
- Notebooks: notebooks/data_exploration.ipynb and notebooks/preprocessing.ipynb for hands-on examples
- Models (placeholders): src\models\spatial_encoder.py and src\models\spatiotemporal_transformer.py

6) Checklist for common tasks (concise)

- Validate dataset before training: python src\preprocessing\preprocess.py
- Inspect train dataset length and a sample tensor: python src\dataset\sequence_dataset.py
- Add training loop: implement src\train.py using the Dataset helpers and models in src\models/

7) AI / assistant integration notes (for Copilot/MCP sessions)

- Key files to open first: src\preprocessing\preprocess.py and src\dataset\sequence_dataset.py. They contain the ground-truth assumptions about labels, image size, and transforms.
- If creating or modifying model code, keep the Dataset output shapes in mind: single-frame samples return an image tensor shaped like (C,H,W); sequence samples return (T,C,H,W) where T==sequence_length. The label is a scalar Long tensor.
- When proposing changes to dataset ingestion, update the validation logic in preprocess.py and the sample check in sequence_dataset.py to ensure backwards compatibility.

---

If this file should incorporate additional repository docs (README, CONTRIBUTING) or other assistant configuration files, provide those paths and text and they will be merged into this guidance.