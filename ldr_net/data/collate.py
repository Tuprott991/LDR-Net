from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch


def detection_collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, Any]]]):
    """Stack fixed-size images and keep targets as a list."""
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


cxr_collate_fn = detection_collate_fn
