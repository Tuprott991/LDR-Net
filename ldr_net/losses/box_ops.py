from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def _box_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor):
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter
    return inter, union


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    inter, union = _box_inter_union(boxes1, boxes2)
    return inter / union.clamp(min=1e-7)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union.clamp(min=1e-7)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-7)
