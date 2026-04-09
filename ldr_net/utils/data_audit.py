from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def summarize_jsonl_samples(samples: Iterable[Mapping]) -> dict:
    num_samples = 0
    empty_box_samples = 0
    disease_supervised_samples = 0
    unique_labels = set()
    images_with_exact_duplicate_boxes = 0
    images_with_conflicting_duplicate_boxes = 0
    max_labels_on_same_box = 0
    box_count_hist = Counter()

    for sample in samples:
        num_samples += 1
        boxes = sample.get("boxes") or []
        labels = sample.get("labels") or []
        disease_labels = sample.get("disease_labels")

        if len(boxes) == 0:
            empty_box_samples += 1
        if disease_labels is not None:
            disease_supervised_samples += 1

        box_count_hist[len(boxes)] += 1
        unique_labels.update(labels)

        grouped = {}
        for box, label in zip(boxes, labels):
            key = tuple(round(float(value), 6) for value in box)
            grouped.setdefault(key, set()).add(int(label))

        if any(len(label_set) > 1 for label_set in grouped.values()):
            images_with_conflicting_duplicate_boxes += 1
        if len(grouped) < len(boxes):
            images_with_exact_duplicate_boxes += 1
        if grouped:
            max_labels_on_same_box = max(max_labels_on_same_box, max(len(label_set) for label_set in grouped.values()))

    max_label = max(unique_labels) if unique_labels else None
    contiguous_labels = unique_labels == set(range(max_label + 1)) if unique_labels else False
    return {
        "num_samples": num_samples,
        "empty_box_samples": empty_box_samples,
        "disease_supervised_samples": disease_supervised_samples,
        "num_unique_labels": len(unique_labels),
        "max_label": max_label,
        "contiguous_labels": contiguous_labels,
        "images_with_exact_duplicate_boxes": images_with_exact_duplicate_boxes,
        "images_with_conflicting_duplicate_boxes": images_with_conflicting_duplicate_boxes,
        "max_labels_on_same_box": max_labels_on_same_box,
        "box_count_hist_top": box_count_hist.most_common(10),
    }


def format_summary(prefix: str, summary: dict) -> str:
    return (
        f"{prefix}: samples={summary['num_samples']} "
        f"empty_boxes={summary['empty_box_samples']} "
        f"disease_supervised={summary['disease_supervised_samples']} "
        f"unique_labels={summary['num_unique_labels']} "
        f"max_label={summary['max_label']} "
        f"exact_dup_images={summary['images_with_exact_duplicate_boxes']} "
        f"conflict_dup_images={summary['images_with_conflicting_duplicate_boxes']}"
    )
