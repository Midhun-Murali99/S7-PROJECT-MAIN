"""
Inference script for single-image classification using the trained ResNet50 model.

Usage examples (from repo root):
  $env:PYTHONPATH='S:\\S7 PROJECT MAIN'; python src\infer.py --image /path/to/image.jpg --model models/best_model.pth --device cpu --topk 3

The script:
- Loads IMAGE_TRANSFORM and CLASS_NAMES from src.preprocessing.preprocess
- Builds a ResNet50 with NUM_CLASSES output units
- Loads model weights from the provided checkpoint (supports raw state_dict or a dict with 'model_state')
- Runs a forward pass and prints the top-k predictions with probabilities
"""
from pathlib import Path
import sys
import argparse

# Ensure repo root on path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from src.classification.resnet50_classifier import TomatoResNet50
from src.preprocessing.preprocess import IMAGE_TRANSFORM, CLASS_NAMES, NUM_CLASSES


def build_model(num_classes, device):
    model = TomatoResNet50(num_classes=num_classes).to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both raw state_dict and dict containers
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    elif isinstance(ckpt, dict) and any(k.startswith('module.') or k in model.state_dict() for k in ckpt.keys()):
        # Assume ckpt is a direct state_dict
        state = ckpt
    else:
        # Fallback: try to load entire dict into model
        state = ckpt

    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Try stripping 'module.' prefixes from state dict keys (common with DataParallel)
        new_state = {}
        for k, v in state.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state[new_key] = v
        model.load_state_dict(new_state)

    return model


def predict_image(model, image_path, device, topk=1):
    image = Image.open(image_path).convert('RGB')
    # Apply transform (expects PIL image)
    input_tensor = IMAGE_TRANSFORM(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    results = []
    for p, idx in zip(top_probs, top_idxs):
        label_name = CLASS_NAMES.get(int(idx), str(idx))
        results.append((int(idx), label_name, float(p)))

    return results


def main(args):
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model checkpoint not found: {model_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    model = build_model(NUM_CLASSES, device)
    model = load_checkpoint(model, model_path, device)

    results = predict_image(model, image_path, device, topk=args.topk)

    print(f"Predictions for: {image_path}")
    for rank, (idx, label, prob) in enumerate(results, start=1):
        print(f"  {rank}. {label} (class {idx}) — probability: {prob:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--topk', type=int, default=1, help='Return top-k predictions')
    args = parser.parse_args()
    main(args)
