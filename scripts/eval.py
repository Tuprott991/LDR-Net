from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from ldr_net.data import CXRDetectionDiseaseDataset, cxr_collate_fn
from ldr_net.engine.trainer import _move_targets_to_device
from ldr_net.losses import HungarianMatcher, LesionDiseaseCriterion
from ldr_net.losses.matcher import box_cxcywh_to_xyxy
from ldr_net.models import LesionDiseaseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LDR-Net detector checkpoint on a JSONL test set.")
    parser.add_argument("--checkpoint", type=str, default="outputs/vindr_stage1/best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--test-json", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test PNG images")
    parser.add_argument("--metadata", type=str, default=None, help="Optional metadata.json with box_classes")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold for precision/recall/F1")
    parser.add_argument("--ap-score-threshold", type=float, default=0.001, help="Low score cutoff before AP calculation")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="Class-aware NMS threshold; use <=0 to disable")
    parser.add_argument("--topk", type=int, default=100, help="Maximum predictions per image before metric matching")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug limit for the number of JSONL samples")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write metrics JSON")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch fallback
        checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model' and 'config' keys.")
    return checkpoint


def load_model(checkpoint: dict[str, Any], device: torch.device) -> tuple[LesionDiseaseNet, dict[str, Any]]:
    model_cfg = checkpoint["config"]["model"]
    model = LesionDiseaseNet(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, model_cfg


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_temp_resolved_jsonl(samples: list[dict[str, Any]], test_dir: Path) -> Path:
    """Resolve image_path against test_dir, falling back to basename when needed."""
    resolved = []
    basename_fallbacks = 0
    missing = 0
    for sample in samples:
        item = dict(sample)
        image_path = Path(str(item["image_path"]))
        if (test_dir / image_path).exists():
            item["image_path"] = str(image_path).replace("\\", "/")
        elif (test_dir / image_path.name).exists():
            item["image_path"] = image_path.name
            basename_fallbacks += 1
        else:
            # Keep the original path so the dataset raises a precise missing-file error.
            item["image_path"] = str(image_path).replace("\\", "/")
            missing += 1
        resolved.append(item)

    temp_path = ROOT / "outputs" / "eval" / "_resolved_test.jsonl"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
        for item in resolved:
            handle.write(json.dumps(item) + "\n")

    if basename_fallbacks:
        print(f"[info] Resolved {basename_fallbacks} image paths by basename inside test_dir.")
    if missing:
        print(f"[warning] {missing} image paths were not found under test_dir and may fail during loading.")
    return temp_path


def load_label_names(num_lesions: int, metadata_arg: str | None) -> tuple[list[str], str]:
    candidate_paths = []
    if metadata_arg is not None:
        candidate_paths.append(Path(metadata_arg))
    candidate_paths.extend(ROOT.glob("data/**/metadata.json"))
    candidate_paths.append(ROOT / "metadata.json")

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        box_classes = metadata.get("box_classes")
        if isinstance(box_classes, list) and len(box_classes) == num_lesions:
            return [str(name) for name in box_classes], f"metadata:{path}"
    return [f"class_{index}" for index in range(num_lesions)], "generic"


def validate_label_space(samples: list[dict[str, Any]], num_lesions: int) -> None:
    max_label = None
    for sample in samples:
        labels = sample.get("labels") or []
        for label in labels:
            label_int = int(label)
            max_label = label_int if max_label is None else max(max_label, label_int)
    if max_label is not None and max_label >= num_lesions:
        raise ValueError(
            f"Test JSONL contains label {max_label}, but the checkpoint only has {num_lesions} lesion classes. "
            "Use the checkpoint trained with this JSONL's label map."
        )


def build_criterion(checkpoint: dict[str, Any], num_lesions: int, device: torch.device) -> LesionDiseaseCriterion | None:
    cfg = checkpoint.get("config", {})
    if "matcher" not in cfg or "loss" not in cfg:
        return None
    matcher = HungarianMatcher(**cfg["matcher"])
    criterion = LesionDiseaseCriterion(num_lesions=num_lesions, matcher=matcher, loss_cfg=cfg["loss"]).to(device)
    criterion.eval()
    return criterion


def postprocess_batch(
    outputs: dict[str, torch.Tensor],
    score_threshold: float,
    nms_threshold: float,
    topk: int,
) -> list[dict[str, torch.Tensor]]:
    probs = outputs["lesion_logits"].softmax(dim=-1)[..., :-1]
    boxes = box_cxcywh_to_xyxy(outputs["lesion_boxes"]).clamp(0.0, 1.0)
    batch_predictions = []

    for batch_index in range(probs.shape[0]):
        scores, labels = probs[batch_index].max(dim=-1)
        keep = scores >= score_threshold
        pred_boxes = boxes[batch_index][keep]
        pred_scores = scores[keep]
        pred_labels = labels[keep]

        if pred_boxes.numel() > 0 and nms_threshold > 0:
            keep_indices = []
            for label in pred_labels.unique():
                label_indices = torch.where(pred_labels == label)[0]
                kept = nms(pred_boxes[label_indices], pred_scores[label_indices], nms_threshold)
                keep_indices.append(label_indices[kept])
            if keep_indices:
                keep_indices = torch.cat(keep_indices)
                pred_boxes = pred_boxes[keep_indices]
                pred_scores = pred_scores[keep_indices]
                pred_labels = pred_labels[keep_indices]

        order = pred_scores.argsort(descending=True)
        if topk > 0:
            order = order[:topk]
        batch_predictions.append(
            {
                "boxes": pred_boxes[order].detach().cpu(),
                "scores": pred_scores[order].detach().cpu(),
                "labels": pred_labels[order].detach().cpu(),
            }
        )
    return batch_predictions


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return float("nan")
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for idx in range(mpre.numel() - 1, 0, -1):
        mpre[idx - 1] = torch.maximum(mpre[idx - 1], mpre[idx])
    changing = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(torch.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]).item())


def compute_ap_for_class(
    detections: list[tuple[int, float, torch.Tensor]],
    targets_by_image: dict[int, torch.Tensor],
    iou_threshold: float,
) -> float:
    num_gt = sum(boxes.shape[0] for boxes in targets_by_image.values())
    if num_gt == 0:
        return float("nan")
    if not detections:
        return 0.0

    detections = sorted(detections, key=lambda item: item[1], reverse=True)
    matched = {image_id: torch.zeros(boxes.shape[0], dtype=torch.bool) for image_id, boxes in targets_by_image.items()}
    tp = torch.zeros(len(detections), dtype=torch.float32)
    fp = torch.zeros(len(detections), dtype=torch.float32)

    for det_index, (image_id, _score, pred_box) in enumerate(detections):
        gt_boxes = targets_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.numel() == 0:
            fp[det_index] = 1.0
            continue
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_index = ious.max(dim=0)
        if best_iou >= iou_threshold and not matched[image_id][best_index]:
            tp[det_index] = 1.0
            matched[image_id][best_index] = True
        else:
            fp[det_index] = 1.0

    recalls = torch.cumsum(tp, dim=0) / max(float(num_gt), 1.0)
    precisions = torch.cumsum(tp, dim=0) / torch.clamp(torch.cumsum(tp + fp, dim=0), min=1.0)
    return average_precision(recalls, precisions)


def compute_precision_recall(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp = 0
    fp = 0
    num_gt = 0

    for prediction, target in zip(predictions, targets):
        pred_keep = prediction["scores"] >= score_threshold
        pred_boxes = prediction["boxes"][pred_keep]
        pred_labels = prediction["labels"][pred_keep]
        pred_scores = prediction["scores"][pred_keep]
        order = pred_scores.argsort(descending=True)

        gt_boxes = box_cxcywh_to_xyxy(target["boxes"]).clamp(0.0, 1.0).cpu()
        gt_labels = target["labels"].cpu()
        num_gt += int(gt_boxes.shape[0])
        matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

        for pred_index in order.tolist():
            label = pred_labels[pred_index]
            class_mask = gt_labels == label
            if class_mask.sum() == 0:
                fp += 1
                continue
            class_indices = torch.where(class_mask)[0]
            ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes[class_indices]).squeeze(0)
            best_iou, best_local_index = ious.max(dim=0)
            best_gt_index = class_indices[best_local_index]
            if best_iou >= iou_threshold and not matched[best_gt_index]:
                tp += 1
                matched[best_gt_index] = True
            else:
                fp += 1

    fn = num_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compute_any_lesion_accuracy(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> float:
    if not targets:
        return float("nan")
    correct = 0
    for prediction, target in zip(predictions, targets):
        pred_has_lesion = bool((prediction["scores"] >= score_threshold).any().item())
        gt_has_lesion = bool(target["boxes"].shape[0] > 0)
        correct += int(pred_has_lesion == gt_has_lesion)
    return correct / len(targets)


def compute_map(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: list[float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    target_boxes_by_class: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    detections_by_class: dict[int, list[tuple[int, float, torch.Tensor]]] = defaultdict(list)

    for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
        gt_boxes = box_cxcywh_to_xyxy(target["boxes"]).clamp(0.0, 1.0).cpu()
        gt_labels = target["labels"].cpu()
        for box, label in zip(gt_boxes, gt_labels):
            target_boxes_by_class[int(label.item())][image_id].append(box)
        for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
            detections_by_class[int(label.item())].append((image_id, float(score.item()), box.cpu()))

    packed_targets: dict[int, dict[int, torch.Tensor]] = {}
    for class_id, image_map in target_boxes_by_class.items():
        packed_targets[class_id] = {
            image_id: torch.stack(boxes, dim=0) if boxes else torch.zeros((0, 4))
            for image_id, boxes in image_map.items()
        }

    per_class: dict[str, dict[str, float]] = {}
    summary: dict[str, float] = {}
    for iou_threshold in iou_thresholds:
        ap_values = []
        for class_id in range(num_classes):
            ap = compute_ap_for_class(
                detections_by_class.get(class_id, []),
                packed_targets.get(class_id, {}),
                iou_threshold=iou_threshold,
            )
            if not torch.isnan(torch.tensor(ap)):
                ap_values.append(ap)
                per_class.setdefault(str(class_id), {})[f"AP@{iou_threshold:.2f}"] = ap
        summary[f"mAP@{iou_threshold:.2f}"] = sum(ap_values) / len(ap_values) if ap_values else float("nan")
    summary["mAP@0.50:0.95"] = (
        sum(summary[f"mAP@{threshold:.2f}"] for threshold in iou_thresholds) / len(iou_thresholds)
        if iou_thresholds
        else float("nan")
    )
    return summary, per_class


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    model, model_cfg = load_model(checkpoint, device)
    label_names, label_source = load_label_names(model_cfg["num_lesions"], args.metadata)

    test_json = Path(args.test_json)
    test_dir = Path(args.test_dir)
    samples = load_jsonl(test_json)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    validate_label_space(samples, model_cfg["num_lesions"])
    resolved_jsonl = write_temp_resolved_jsonl(samples, test_dir)

    dataset = CXRDetectionDiseaseDataset(
        annotations_path=resolved_jsonl,
        image_root=test_dir,
        image_size=model_cfg["image_size"],
        deduplicate_boxes=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=cxr_collate_fn,
    )
    criterion = build_criterion(checkpoint, model_cfg["num_lesions"], device)

    all_predictions: list[dict[str, torch.Tensor]] = []
    all_targets: list[dict[str, torch.Tensor]] = []
    loss_totals: dict[str, float] = defaultdict(float)
    steps = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            device_targets = _move_targets_to_device(targets, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                if criterion is not None:
                    loss_dict = criterion(outputs, device_targets)
            batch_predictions = postprocess_batch(
                outputs=outputs,
                score_threshold=args.ap_score_threshold,
                nms_threshold=args.nms_threshold,
                topk=args.topk,
            )
            all_predictions.extend(batch_predictions)
            all_targets.extend(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
                for target in device_targets
            )
            if criterion is not None:
                for key, value in loss_dict.items():
                    loss_totals[key] += float(value.detach().item())
            steps += 1

    iou_thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    pr = compute_precision_recall(all_predictions, all_targets, args.score_threshold, iou_threshold=0.5)
    any_lesion_accuracy = compute_any_lesion_accuracy(all_predictions, all_targets, args.score_threshold)
    map_summary, per_class = compute_map(all_predictions, all_targets, model_cfg["num_lesions"], iou_thresholds)

    gt_boxes = sum(int(target["boxes"].shape[0]) for target in all_targets)
    pred_boxes = sum(int((prediction["scores"] >= args.score_threshold).sum().item()) for prediction in all_predictions)
    metrics: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint)),
        "test_json": str(test_json),
        "test_dir": str(test_dir),
        "num_images": len(dataset),
        "num_lesions": model_cfg["num_lesions"],
        "label_source": label_source,
        "gt_boxes": gt_boxes,
        "pred_boxes_at_score_threshold": pred_boxes,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "any_lesion_accuracy": any_lesion_accuracy,
        **pr,
        **map_summary,
        "losses": {key: value / max(steps, 1) for key, value in loss_totals.items()},
        "per_class": per_class,
    }

    print(f"checkpoint: {metrics['checkpoint']}")
    print(f"test_json: {metrics['test_json']}")
    print(f"test_dir: {metrics['test_dir']}")
    print(f"num_images: {metrics['num_images']}")
    print(f"num_lesions: {metrics['num_lesions']}")
    print(f"label_source: {metrics['label_source']}")
    if label_source == "generic":
        print("warning: no matching metadata.json was found, so per-class names are generic")
    print(f"gt_boxes: {gt_boxes}")
    print(f"pred_boxes@score>={args.score_threshold:.3f}: {pred_boxes}")
    print(f"any_lesion_accuracy@score>={args.score_threshold:.3f}: {metrics['any_lesion_accuracy']:.4f}")
    print(
        "precision@0.50: "
        f"{metrics['precision']:.4f}  recall@0.50: {metrics['recall']:.4f}  f1@0.50: {metrics['f1']:.4f}"
    )
    print(f"mAP@0.50: {metrics['mAP@0.50']:.4f}")
    print(f"mAP@0.50:0.95: {metrics['mAP@0.50:0.95']:.4f}")
    if metrics["losses"]:
        print("losses:")
        for key, value in metrics["losses"].items():
            print(f"  {key}: {value:.4f}")

    class_ap50 = []
    for class_id, values in per_class.items():
        ap50 = values.get("AP@0.50")
        if ap50 is not None:
            class_ap50.append((int(class_id), ap50))
    if class_ap50:
        print("per_class_AP@0.50:")
        for class_id, ap50 in sorted(class_ap50, key=lambda item: item[1], reverse=True):
            name = label_names[class_id] if class_id < len(label_names) else f"class_{class_id}"
            print(f"  {class_id:02d} {name}: {ap50:.4f}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
