# Project Structure

This project now separates the active classification pipeline from the older detection experiments.

## Active pipeline

- src/classification/train.py
- src/classification/resnet50_classifier.py
- src/classification/__init__.py

This is the current recommended training path for tomato disease classification using ResNet-50.

## Legacy / experimental code

The older YOLO-style detection code is still present for reference and comparison, but it is no longer the primary project workflow.

- src/train_complete.py
- src/models/yolov8_detector.py
- src/preprocessing/preprocess_complete.py

## Notes

- The classification path is simpler and easier to maintain.
- The legacy detection path remains available if you want to continue object detection research.
- Use the classification training script unless you specifically need the detector.
