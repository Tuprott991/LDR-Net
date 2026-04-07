import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from ldr_net.data import CXRDetectionDiseaseDataset, SyntheticCXRDataset, cxr_collate_fn
from ldr_net.engine import evaluate, train_one_epoch
from ldr_net.losses import HungarianMatcher, LesionDiseaseCriterion
from ldr_net.models import LesionDiseaseNet
from ldr_net.utils.checkpoint import save_checkpoint
from ldr_net.utils.config import load_config
from ldr_net.utils.train import build_optimizer, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train LDR-Net")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training max steps")
    return parser.parse_args()


def build_datasets(cfg, synthetic=False):
    data_cfg = cfg["data"]
    if synthetic:
        train_dataset = SyntheticCXRDataset(
            num_samples=data_cfg["synthetic_train_samples"],
            image_size=data_cfg["image_size"],
            num_lesions=cfg["model"]["num_lesions"],
            num_diseases=cfg["model"]["num_diseases"],
            max_boxes=data_cfg["max_boxes_per_sample"],
        )
        val_dataset = SyntheticCXRDataset(
            num_samples=data_cfg["synthetic_val_samples"],
            image_size=data_cfg["image_size"],
            num_lesions=cfg["model"]["num_lesions"],
            num_diseases=cfg["model"]["num_diseases"],
            max_boxes=data_cfg["max_boxes_per_sample"],
        )
        return train_dataset, val_dataset

    train_dataset = CXRDetectionDiseaseDataset(
        annotations_path=data_cfg["train_json"],
        image_root=data_cfg["image_root"],
        image_size=data_cfg["image_size"],
    )
    val_dataset = CXRDetectionDiseaseDataset(
        annotations_path=data_cfg["val_json"],
        image_root=data_cfg["image_root"],
        image_size=data_cfg["image_size"],
    )
    return train_dataset, val_dataset


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    device = torch.device(cfg["training"]["device"])
    seed_everything(cfg["training"]["seed"])

    model = LesionDiseaseNet(**cfg["model"]).to(device)
    matcher = HungarianMatcher(**cfg["matcher"])
    criterion = LesionDiseaseCriterion(
        num_lesions=cfg["model"]["num_lesions"],
        matcher=matcher,
        loss_cfg=cfg["loss"],
    ).to(device)
    optimizer = build_optimizer(model, cfg["optimizer"])

    train_dataset, val_dataset = build_datasets(cfg, synthetic=args.synthetic)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=cxr_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=cxr_collate_fn,
    )

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every=cfg["training"]["log_every"],
            max_steps=cfg["training"]["max_steps"],
            grad_clip_norm=cfg["training"]["grad_clip_norm"],
            mixed_precision=cfg["training"]["mixed_precision"],
        )
        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            max_steps=cfg["training"]["max_steps"],
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(checkpoint, output_dir / "last.pt")

        current_val = val_metrics.get("loss_total", float("inf"))
        if current_val <= best_val:
            best_val = current_val
            save_checkpoint(checkpoint, output_dir / "best.pt")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.get('loss_total', float('nan')):.4f} "
            f"val_loss={current_val:.4f}"
        )


if __name__ == "__main__":
    main()
