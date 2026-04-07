from __future__ import annotations

from typing import Dict, Mapping

import torch


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _reduce_loss_dict(loss_dict: Mapping[str, torch.Tensor], weight_dict: Mapping[str, float]):
    total = None
    detached = {}
    for name, loss in loss_dict.items():
        if not isinstance(loss, torch.Tensor):
            continue
        if loss.ndim != 0:
            loss = loss.mean()
        detached[name] = loss.detach()
        if name in weight_dict:
            total = loss * weight_dict[name] if total is None else total + loss * weight_dict[name]
    if total is None:
        tensor = next((v for v in loss_dict.values() if isinstance(v, torch.Tensor)), None)
        if tensor is None:
            total = torch.tensor(0.0)
        else:
            total = tensor.sum() * 0.0
    return total, detached


def _accumulate(metrics: Dict[str, float], loss_dict: Mapping[str, torch.Tensor], batch_size: int):
    for key, value in loss_dict.items():
        metrics[key] = metrics.get(key, 0.0) + float(value) * batch_size


def _finalize(metrics: Dict[str, float], count: int):
    return {key: value / max(count, 1) for key, value in metrics.items()}


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch: int = 0,
    max_norm: float = 0.0,
):
    model.train()
    criterion.train()

    metric_sums: Dict[str, float] = {}
    num_samples = 0
    num_batches = 0
    weight_dict = getattr(criterion, "weight_dict", {})

    for images, targets in data_loader:
        images = [_move_to_device(image, device) for image in images]
        targets = [_move_to_device(target, device) for target in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        total_loss, detached_losses = _reduce_loss_dict(loss_dict, weight_dict)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        batch_size = len(images)
        num_samples += batch_size
        num_batches += 1
        _accumulate(metric_sums, detached_losses, batch_size)
        metric_sums["loss"] = metric_sums.get("loss", 0.0) + float(total_loss.detach()) * batch_size

    metrics = _finalize(metric_sums, num_samples)
    metrics["epoch"] = epoch
    metrics["num_batches"] = num_batches
    return metrics


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_sums: Dict[str, float] = {}
    num_samples = 0
    num_batches = 0
    weight_dict = getattr(criterion, "weight_dict", {})

    for images, targets in data_loader:
        images = [_move_to_device(image, device) for image in images]
        targets = [_move_to_device(target, device) for target in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        total_loss, detached_losses = _reduce_loss_dict(loss_dict, weight_dict)

        batch_size = len(images)
        num_samples += batch_size
        num_batches += 1
        _accumulate(metric_sums, detached_losses, batch_size)
        metric_sums["loss"] = metric_sums.get("loss", 0.0) + float(total_loss.detach()) * batch_size

    metrics = _finalize(metric_sums, num_samples)
    metrics["num_batches"] = num_batches
    return metrics
