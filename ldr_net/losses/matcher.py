from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _is_box_supervised(target: Dict) -> bool:
    return bool(target.get("has_boxes", target.get("box_annotated", True)))


def _greedy_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[int] = []
    cols: List[int] = []
    n_rows, n_cols = cost.shape
    used_rows = set()
    used_cols = set()
    for _ in range(min(n_rows, n_cols)):
        best = None
        best_value = None
        for r in range(n_rows):
            if r in used_rows:
                continue
            for c in range(n_cols):
                if c in used_cols:
                    continue
                value = cost[r, c].item()
                if best_value is None or value < best_value:
                    best = (r, c)
                    best_value = value
        if best is None:
            break
        r, c = best
        used_rows.add(r)
        used_cols.add(c)
        rows.append(r)
        cols.append(c)
    device = cost.device
    return torch.as_tensor(rows, dtype=torch.int64, device=device), torch.as_tensor(cols, dtype=torch.int64, device=device)


class HungarianMatcher(nn.Module):
    """Match queries and targets using the Hungarian algorithm."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All costs cannot be zero.")

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Sequence[Dict]):
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if pred_logits is None or pred_boxes is None:
            raise KeyError("HungarianMatcher expects 'pred_logits' and 'pred_boxes' in model outputs.")

        out_prob = pred_logits.softmax(-1)
        out_bbox = pred_boxes
        indices = []
        for batch_index, target in enumerate(targets):
            if not _is_box_supervised(target):
                empty = torch.empty(0, dtype=torch.int64, device=out_bbox.device)
                indices.append((empty, empty))
                continue

            tgt_ids = target.get("labels")
            tgt_bbox = target.get("boxes")
            if tgt_ids is None or tgt_bbox is None or tgt_bbox.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=out_bbox.device)
                indices.append((empty, empty))
                continue

            tgt_ids = tgt_ids.to(device=out_bbox.device, dtype=torch.int64)
            tgt_bbox = tgt_bbox.to(device=out_bbox.device, dtype=out_bbox.dtype)

            if self.num_classes is not None:
                prob = out_prob[batch_index, :, :-1] if pred_logits.shape[-1] == self.num_classes + 1 else out_prob[batch_index]
            else:
                prob = out_prob[batch_index, :, :-1] if pred_logits.shape[-1] > int(tgt_ids.max().item()) + 1 else out_prob[batch_index]

            cost_class = -prob[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[batch_index], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[batch_index]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            cost = cost.detach().cpu()
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
                row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=out_bbox.device)
                col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=out_bbox.device)
            else:  # pragma: no cover
                row_ind, col_ind = _greedy_assignment(cost.to(out_bbox.device))
            indices.append((row_ind, col_ind))
        return indices
