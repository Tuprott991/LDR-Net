from __future__ import annotations

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    c_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    c_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[..., 0] * c_wh[..., 1]

    return iou - (c_area - union) / c_area.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    """Match predicted lesion queries to target lesions."""

    def __init__(self, class_cost: float = 2.0, bbox_cost: float = 5.0, giou_cost: float = 2.0) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[Tensor, Tensor]]:
        pred_logits = outputs["lesion_logits"]
        pred_boxes = outputs["lesion_boxes"]
        batch_size = pred_logits.shape[0]
        indices: List[Tuple[Tensor, Tensor]] = []

        for batch_index in range(batch_size):
            target = targets[batch_index]
            if not bool(target.get("has_boxes", True)) or target["boxes"].numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            out_prob = pred_logits[batch_index].softmax(dim=-1)
            out_bbox = pred_boxes[batch_index]
            tgt_ids = target["labels"]
            tgt_bbox = target["boxes"]

            cost_class = -out_prob[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            cost_matrix = (
                self.class_cost * cost_class
                + self.bbox_cost * cost_bbox
                + self.giou_cost * cost_giou
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=pred_logits.device),
                )
            )

        return indices
