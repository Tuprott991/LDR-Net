import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms.functional import pil_to_tensor

from ldr_net.models import LesionDiseaseNet

DEFAULT_VINDR_40_LABELS = [
    "Boot-shaped heart",
    "Peribronchovascular interstitial opacity",
    "Reticulonodular opacity",
    "Bronchial thickening",
    "Enlarged PA",
    "Cardiomegaly",
    "Diffuse aveolar opacity",
    "Other opacity",
    "Other lesion",
    "Consolidation",
    "Mediastinal shift",
    "Anterior mediastinal mass",
    "Dextro cardia",
    "Pleural effusion",
    "Stomach on the right side",
    "Atelectasis",
    "Lung hyperinflation",
    "Egg on string sign",
    "Interstitial lung disease - ILD",
    "Infiltration",
    "Lung cavity",
    "Pneumothorax",
    "Edema",
    "Pleural thickening",
    "Other nodule/mass",
    "Clavicle fracture",
    "Chest wall mass",
    "Lung cyst",
    "Emphysema",
    "Bronchectasis",
    "Expanded edges of the anterior ribs",
    "Pulmonary fibrosis",
    "Paraveterbral mass",
    "Aortic enlargement",
    "Calcification",
    "Intrathoracic digestive structure",
    "ILD",
    "Nodule/Mass",
    "Lung Opacity",
    "Rib fracture",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDR-Net inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, default="outputs/vindr_stage1/best.pt", help="Checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--metadata", type=str, default=None, help="Optional metadata.json path with box_classes")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Minimum detection confidence")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="Class-agnostic NMS IoU threshold")
    parser.add_argument("--topk", type=int, default=20, help="Maximum number of detections to draw")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Make sure training finished and produced a real .pt file."
        )
    if path.suffix == ".crdownload":
        raise RuntimeError(f"Checkpoint is incomplete: {path}")
    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model' and 'config' keys.")
    return checkpoint


def load_model_from_checkpoint(checkpoint: dict, device: torch.device):
    model_cfg = checkpoint["config"]["model"]
    model = LesionDiseaseNet(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, model_cfg


def load_label_names(num_lesions: int, metadata_arg: str | None):
    candidate_paths = []
    if metadata_arg is not None:
        candidate_paths.append(Path(metadata_arg))
    candidate_paths.extend(ROOT.glob("data/**/metadata.json"))

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        box_classes = metadata.get("box_classes")
        if isinstance(box_classes, list) and len(box_classes) == num_lesions:
            return box_classes

    if num_lesions == len(DEFAULT_VINDR_40_LABELS):
        return DEFAULT_VINDR_40_LABELS

    return [f"class_{index}" for index in range(num_lesions)]


def load_image_tensor(image_path: Path, image_size: int):
    image = Image.open(image_path).convert("L")
    original_size = image.size
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    tensor = pil_to_tensor(resized).float() / 255.0
    return image, tensor, original_size


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def postprocess(outputs: dict, image_width: int, image_height: int, score_threshold: float, nms_threshold: float, topk: int):
    logits = outputs["lesion_logits"][0]
    boxes = outputs["lesion_boxes"][0]

    probs = logits.softmax(dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)
    keep = scores >= score_threshold

    if keep.sum() == 0:
        return [], torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    boxes = cxcywh_to_xyxy(boxes)
    boxes[:, [0, 2]] *= image_width
    boxes[:, [1, 3]] *= image_height
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_height - 1)

    if boxes.shape[0] > 0 and nms_threshold > 0:
        keep_idx = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]

    if boxes.shape[0] > topk:
        order = scores.argsort(descending=True)[:topk]
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
    else:
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]

    return order, boxes, labels, scores


def draw_predictions(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, label_names):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        color = (255, 64, 64)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = f"{label_names[label]} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), caption, fill=(255, 255, 255), font=font)

    return image


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, model_cfg = load_model_from_checkpoint(checkpoint, device)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_image, image_tensor, (image_width, image_height) = load_image_tensor(
        image_path,
        image_size=model_cfg["image_size"],
    )
    label_names = load_label_names(model_cfg["num_lesions"], args.metadata)

    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))

    _, boxes, labels, scores = postprocess(
        outputs=outputs,
        image_width=image_width,
        image_height=image_height,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        topk=args.topk,
    )

    annotated = draw_predictions(original_image, boxes, labels, scores, label_names)

    output_path = Path(args.output) if args.output else ROOT / "outputs" / "inference" / f"{image_path.stem}_pred.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)

    print(f"checkpoint: {checkpoint_path}")
    print(f"image: {image_path}")
    print(f"detections: {len(scores)}")
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        print(f"{label_names[label]}\tscore={score:.4f}\tbox={[round(v, 2) for v in box]}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
