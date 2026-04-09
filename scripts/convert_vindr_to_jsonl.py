import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_DISEASE_COLUMNS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]
EXCLUDED_BOX_CLASSES = {"No finding"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VinDr-CXR CSVs into LDR-Net JSONL files")
    parser.add_argument("--source-root", type=str, required=True, help="Folder containing train/test dirs and VinDr CSVs")
    parser.add_argument("--output-dir", type=str, default="data/vindr_physionet_384", help="Output folder for JSONL and metadata")
    parser.add_argument("--image-ext", type=str, default=".png", help="Image file extension")
    parser.add_argument("--positive-vote-threshold", type=int, default=1, help="Minimum radiologist votes required for a positive label")
    parser.add_argument("--train-dir", type=str, default="train", help="Relative train image dir under source root")
    parser.add_argument("--test-dir", type=str, default="test", help="Relative test image dir under source root")
    parser.add_argument("--annotation-train-csv", type=str, default=None, help="Optional train annotation CSV path relative to source root")
    parser.add_argument("--annotation-test-csv", type=str, default=None, help="Optional test annotation CSV path relative to source root")
    parser.add_argument("--label-train-csv", type=str, default=None, help="Optional train image-level label CSV path relative to source root")
    parser.add_argument("--label-test-csv", type=str, default=None, help="Optional test image-level label CSV path relative to source root")
    return parser.parse_args()


def load_image_size(path: Path):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        from torchvision.io import read_image

        image = read_image(str(path))
        return int(image.shape[-2]), int(image.shape[-1])

    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def read_label_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    return columns, rows


def empty_label_info():
    return {
        "disease_labels": None,
        "lesion_concept_labels": None,
    }


def list_image_ids(image_dir: Path, image_ext: str):
    image_ids = []
    for path in sorted(image_dir.glob(f"*{image_ext}")):
        image_ids.append(path.stem)
    return image_ids


def aggregate_image_labels(rows, disease_columns, positive_vote_threshold):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)

    all_columns = [key for key in rows[0].keys() if key not in {"image_id", "rad_id", "rad_ID"}]
    lesion_concept_columns = [column for column in all_columns if column not in disease_columns]

    aggregated = {}
    for image_id, image_rows in grouped.items():
        disease_labels = []
        lesion_concept_labels = []
        for column in disease_columns:
            votes = sum(int(row[column]) for row in image_rows)
            disease_labels.append(1 if votes >= positive_vote_threshold else 0)
        for column in lesion_concept_columns:
            votes = sum(int(row[column]) for row in image_rows)
            lesion_concept_labels.append(1 if votes >= positive_vote_threshold else 0)
        aggregated[image_id] = {
            "disease_labels": disease_labels,
            "lesion_concept_labels": lesion_concept_labels,
        }

    return aggregated, lesion_concept_columns


def collect_box_class_names(annotation_csv_paths):
    class_names = []
    seen = set()
    for annotation_csv_path in annotation_csv_paths:
        with annotation_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                class_name = row["class_name"]
                if class_name in EXCLUDED_BOX_CLASSES:
                    continue
                if class_name not in seen:
                    seen.add(class_name)
                    class_names.append(class_name)
    return class_names


def read_annotations(annotation_csv_path: Path, class_to_index: dict):
    grouped = defaultdict(list)
    with annotation_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["class_name"] in EXCLUDED_BOX_CLASSES:
                continue
            if not row["x_min"] or not row["y_min"] or not row["x_max"] or not row["y_max"]:
                continue
            grouped[row["image_id"]].append(
                {
                    "class_name": row["class_name"],
                    "label": class_to_index[row["class_name"]],
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                    "rad_id": row.get("rad_id", row.get("rad_ID")),
                }
            )
    return grouped


def make_record(image_id, split_dir, image_ext, image_root, label_info, annotations):
    image_rel_path = Path(split_dir) / f"{image_id}{image_ext}"
    image_path = image_root / image_rel_path
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    image_height, image_width = load_image_size(image_path)
    boxes = []
    labels = []
    annotation_rows = annotations.get(image_id, [])
    out_of_bounds = 0
    for row in annotation_rows:
        x_min = max(0.0, min(row["x_min"], image_width))
        y_min = max(0.0, min(row["y_min"], image_height))
        x_max = max(0.0, min(row["x_max"], image_width))
        y_max = max(0.0, min(row["y_max"], image_height))
        if x_max <= x_min or y_max <= y_min:
            out_of_bounds += 1
            continue

        cx = ((x_min + x_max) / 2.0) / image_width
        cy = ((y_min + y_max) / 2.0) / image_height
        w = (x_max - x_min) / image_width
        h = (y_max - y_min) / image_height
        boxes.append([cx, cy, w, h])
        labels.append(int(row["label"]))

    return {
        "image_path": str(image_rel_path).replace("\\", "/"),
        "sample_id": image_id,
        "boxes": boxes,
        "labels": labels,
        "disease_labels": label_info["disease_labels"],
        "lesion_concept_labels": label_info["lesion_concept_labels"],
        "has_detection": True,
        "has_disease_labels": label_info["disease_labels"] is not None,
        "num_skipped_boxes": out_of_bounds,
    }


def convert_split(
    source_root: Path,
    split_name: str,
    image_dir_name: str,
    output_dir: Path,
    disease_columns,
    positive_vote_threshold,
    image_ext,
    class_to_index,
    annotation_csv: Path,
    label_csv: Path | None,
):
    image_dir = source_root / image_dir_name
    if label_csv is not None and label_csv.exists():
        _, label_rows = read_label_rows(label_csv)
        aggregated_labels, lesion_concept_columns = aggregate_image_labels(
            label_rows,
            disease_columns=disease_columns,
            positive_vote_threshold=positive_vote_threshold,
        )
        image_ids = sorted(set(aggregated_labels.keys()))
    else:
        aggregated_labels = defaultdict(empty_label_info)
        lesion_concept_columns = []
        image_ids = list_image_ids(image_dir=image_dir, image_ext=image_ext)
    annotations = read_annotations(annotation_csv, class_to_index=class_to_index)
    image_ids = sorted(set(image_ids) | set(annotations.keys()))

    records = []
    for image_id in image_ids:
        label_info = aggregated_labels[image_id]
        records.append(
            make_record(
                image_id=image_id,
                split_dir=image_dir_name,
                image_ext=image_ext,
                image_root=source_root,
                label_info=label_info,
                annotations=annotations,
            )
        )

    output_path = output_dir / f"{split_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return output_path, lesion_concept_columns, len(records)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    train_annotation_csv = source_root / (args.annotation_train_csv or "annotations_train.csv")
    test_annotation_csv = source_root / (args.annotation_test_csv or "annotations_test.csv")
    train_label_csv = source_root / args.label_train_csv if args.label_train_csv else source_root / "image_labels_train.csv"
    test_label_csv = source_root / args.label_test_csv if args.label_test_csv else source_root / "image_labels_test.csv"

    box_class_names = collect_box_class_names(
        [
            train_annotation_csv,
            test_annotation_csv,
        ]
    )
    class_to_index = {name: index for index, name in enumerate(box_class_names)}

    train_jsonl, lesion_concept_columns, train_count = convert_split(
        source_root=source_root,
        split_name="train",
        image_dir_name=args.train_dir,
        output_dir=output_dir,
        disease_columns=DEFAULT_DISEASE_COLUMNS,
        positive_vote_threshold=args.positive_vote_threshold,
        image_ext=args.image_ext,
        class_to_index=class_to_index,
        annotation_csv=train_annotation_csv,
        label_csv=train_label_csv if train_label_csv.exists() else None,
    )
    test_jsonl, _, test_count = convert_split(
        source_root=source_root,
        split_name="test",
        image_dir_name=args.test_dir,
        output_dir=output_dir,
        disease_columns=DEFAULT_DISEASE_COLUMNS,
        positive_vote_threshold=args.positive_vote_threshold,
        image_ext=args.image_ext,
        class_to_index=class_to_index,
        annotation_csv=test_annotation_csv,
        label_csv=test_label_csv if test_label_csv.exists() else None,
    )

    metadata = {
        "source_root": str(source_root),
        "train_jsonl": str(train_jsonl),
        "test_jsonl": str(test_jsonl),
        "image_extension": args.image_ext,
        "positive_vote_threshold": args.positive_vote_threshold,
        "box_classes": box_class_names,
        "lesion_concept_classes": lesion_concept_columns,
        "disease_classes": DEFAULT_DISEASE_COLUMNS if train_label_csv.exists() or test_label_csv.exists() else [],
        "train_samples": train_count,
        "test_samples": test_count,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {train_count} train samples to {train_jsonl}")
    print(f"Wrote {test_count} test samples to {test_jsonl}")
    print(f"Wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
