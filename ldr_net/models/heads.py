"""Heads and token builders for the LDR-Net skeleton."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiScaleProjector(nn.Module):
    """Fuse multi-scale features into a single high-resolution tensor."""

    def __init__(self, in_channels: int, out_channels: int = 256, levels: int = 4) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
            for _ in range(levels)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * levels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        fused = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False)
            fused.append(projected)
        return self.fuse(torch.cat(fused, dim=1))


class LesionDecoder(nn.Module):
    """Transformer decoder that proposes lesion queries from fused features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 12,
        num_heads: int = 8,
        num_layers: int = 2,
        num_lesion_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_lesion_classes = num_lesion_classes
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.class_head = nn.Linear(hidden_dim, num_lesion_classes + 1)
        self.box_head = nn.Linear(hidden_dim, 4)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, fused_map: Tensor) -> dict[str, Tensor]:
        bsz, _, _, _ = fused_map.shape
        memory = self.input_proj(fused_map).flatten(2).transpose(1, 2)
        queries = self.query_embed.weight.unsqueeze(0).expand(bsz, -1, -1)
        hidden = self.decoder(queries, memory)

        lesion_logits = self.class_head(hidden)
        lesion_boxes = torch.sigmoid(self.box_head(hidden))
        lesion_scores = lesion_logits[..., :-1].sigmoid().amax(dim=-1)
        lesion_uncertainty = F.softplus(self.uncertainty_head(hidden)).squeeze(-1)

        return {
            "lesion_logits": lesion_logits,
            "lesion_boxes": lesion_boxes,
            "lesion_uncertainty": lesion_uncertainty,
            "lesion_hidden": hidden,
            "lesion_mask": lesion_scores,
        }


class LesionTokenBuilder(nn.Module):
    """Select and enrich the highest-confidence lesion queries."""

    def __init__(self, hidden_dim: int = 256, topk: int = 20) -> None:
        super().__init__()
        self.topk = topk
        self.box_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, lesion_hidden: Tensor, lesion_boxes: Tensor, lesion_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        lesion_scores = lesion_logits[..., :-1].sigmoid().amax(dim=-1)
        topk = min(self.topk, lesion_hidden.shape[1])
        topk_scores, topk_indices = torch.topk(lesion_scores, k=topk, dim=1)

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, lesion_hidden.shape[-1])
        topk_hidden = lesion_hidden.gather(1, gather_index)
        topk_boxes = lesion_boxes.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, lesion_boxes.shape[-1]))

        score_tokens = self.score_embed(topk_scores.unsqueeze(-1))
        tokens = topk_hidden + self.box_embed(topk_boxes) + score_tokens
        return self.out_norm(tokens), topk_scores, topk_indices


class AnatomyBranch(nn.Module):
    """Produce a compact set of anatomy-aware tokens from the fused map."""

    def __init__(self, hidden_dim: int = 256, num_tokens: int = 8) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.pool_shape: Tuple[int, int] = (2, max(1, num_tokens // 2))
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, fused_map: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(self.proj(fused_map), self.pool_shape)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.norm(tokens[:, : self.num_tokens, :])


class GlobalTokenHead(nn.Module):
    """Encode the global image context as a single token."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, fused_map: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(fused_map, output_size=1).flatten(1)
        token = self.net(pooled)
        return self.norm(token).unsqueeze(1)


class PrototypeBank(nn.Module):
    """Learned lesion prototypes used as a reasoning prior."""

    def __init__(self, hidden_dim: int = 256, num_prototypes: int = 16) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim) * 0.02)
        self.scale = hidden_dim**-0.5

    def forward(self, lesion_tokens: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(lesion_tokens @ self.prototypes.t() * self.scale, dim=-1)
        context = weights @ self.prototypes
        return context, weights


class DiseaseReasoner(nn.Module):
    """Combine lesion, anatomy, and prototype context into reasoning tokens."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 8) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        lesion_tokens: Tensor,
        anatomy_tokens: Tensor,
        global_token: Tensor,
        proto_context: Tensor,
    ) -> Tensor:
        tokens = torch.cat([global_token, anatomy_tokens, lesion_tokens + proto_context], dim=1)
        return self.norm(self.encoder(tokens))


class DiseaseHead(nn.Module):
    """Final multi-label disease classifier."""

    def __init__(self, hidden_dim: int = 256, num_diseases: int = 14) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim * 2, num_diseases),
        )

    def forward(self, z_reason: Tensor) -> Tensor:
        cls_token = z_reason[:, 0]
        attn = torch.softmax(self.attn(z_reason).squeeze(-1), dim=-1)
        pooled = torch.einsum("bn,bnc->bc", attn, z_reason)
        return self.head(torch.cat([cls_token, pooled], dim=-1))
