from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn


def _move_targets_to_device(targets: List[dict], device: torch.device) -> List[dict]:
    moved = []
    for target in targets:
        moved_target = {}
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value
        moved.append(moved_target)
    return moved


def _mean_metrics(totals: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps == 0:
        return {}
    return {key: value / steps for key, value in totals.items()}


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 10,
    max_steps: int | None = None,
    grad_clip_norm: float | None = None,
    mixed_precision: bool = False,
) -> Dict[str, float]:
    del mixed_precision
    model.train()
    if getattr(model, "_freeze_backbone_active", False):
        model.backbone.eval()
    criterion.train()
    totals = defaultdict(float)
    disease_weight = 1.0
    if hasattr(criterion, "loss_cfg"):
        disease_weight = float(criterion.loss_cfg.get("disease_weight", 1.0))

    for step, (images, targets) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device)
        targets = _move_targets_to_device(targets, device)

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss_total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        for key, value in loss_dict.items():
            totals[key] += float(value.detach().item())

        if step % log_every == 0:
            message = (
                f"epoch={epoch} step={step} "
                f"loss={loss_dict['loss_total'].detach().item():.4f} "
                f"detection={loss_dict['loss_detection'].detach().item():.4f}"
            )
            if disease_weight != 0.0:
                message += f" disease={loss_dict['loss_disease'].detach().item():.4f}"
            print(message)

    return _mean_metrics(totals, max(1, min(len(dataloader), (max_steps or len(dataloader)))))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()
    totals = defaultdict(float)
    steps = 0

    for step, (images, targets) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device)
        targets = _move_targets_to_device(targets, device)

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        for key, value in loss_dict.items():
            totals[key] += float(value.detach().item())
        steps += 1

    return _mean_metrics(totals, steps)
