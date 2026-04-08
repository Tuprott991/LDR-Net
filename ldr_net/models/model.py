"""Top-level lesion-aware chest X-ray model."""

from __future__ import annotations

from torch import Tensor, nn

from .backbone import HFConvNeXtV2Backbone, build_backbone
from .fpn import FeaturePyramidNetwork
from .heads import (
    AnatomyBranch,
    DiseaseHead,
    DiseaseReasoner,
    GlobalTokenHead,
    LesionDecoder,
    LesionTokenBuilder,
    MultiScaleProjector,
    PrototypeBank,
)


class LesionDiseaseNet(nn.Module):
    """End-to-end lesion-aware CXR model skeleton.

    The module is intentionally compact and dependency-light so that the
    training subsystem can attach custom losses, matching, or supervision
    without changing the forward contract.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 1024,
        backbone_type: str = "hf_convnextv2",
        backbone_name: str = HFConvNeXtV2Backbone.DEFAULT_MODEL_NAME,
        backbone_local_files_only: bool = False,
        backbone_fallback_to_resnet: bool = True,
        num_lesions: int = 22,
        num_diseases: int = 14,
        num_queries: int = 100,
        dim: int = 256,
        num_anatomy_tokens: int = 8,
        num_prototypes: int = 32,
        lesion_topk: int = 20,
        decoder_layers: int = 4,
        reasoner_layers: int = 3,
        nhead: int = 8,
        **_: dict,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            backbone_name=backbone_name,
            in_channels=in_channels,
            local_files_only=backbone_local_files_only,
            fallback_to_resnet=backbone_fallback_to_resnet,
        )
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=dim)
        self.projector = MultiScaleProjector(dim, out_channels=dim, levels=4)
        self.lesion_decoder = LesionDecoder(
            hidden_dim=dim,
            num_queries=num_queries,
            num_heads=nhead,
            num_layers=decoder_layers,
            num_lesion_classes=num_lesions,
        )
        self.lesion_token_builder = LesionTokenBuilder(hidden_dim=dim, topk=lesion_topk)
        self.anatomy_branch = AnatomyBranch(hidden_dim=dim, num_tokens=num_anatomy_tokens)
        self.global_token_head = GlobalTokenHead(hidden_dim=dim)
        self.prototype_bank = PrototypeBank(hidden_dim=dim, num_prototypes=num_prototypes)
        self.reasoner = DiseaseReasoner(hidden_dim=dim, num_layers=reasoner_layers, num_heads=nhead)
        self.disease_head = DiseaseHead(hidden_dim=dim, num_diseases=num_diseases)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        pyramids = self.fpn(features)
        fused_map = self.projector(list(pyramids.values()))

        lesion_state = self.lesion_decoder(fused_map)
        lesion_tokens = self.lesion_token_builder(
            lesion_state["lesion_hidden"], lesion_state["lesion_boxes"], lesion_state["lesion_logits"]
        )
        selected_tokens, selected_scores, topk_idx = lesion_tokens
        anatomy_tokens = self.anatomy_branch(fused_map)
        global_token = self.global_token_head(fused_map)
        proto_context, proto_weights = self.prototype_bank(selected_tokens)
        z_reason = self.reasoner(selected_tokens, anatomy_tokens, global_token, proto_context)
        disease_logits = self.disease_head(z_reason)

        return {
            **lesion_state,
            "lesion_tokens": selected_tokens,
            "lesion_mask": selected_scores,
            "topk_idx": topk_idx,
            "anatomy_tokens": anatomy_tokens,
            "global_token": global_token,
            "proto_context": proto_context,
            "proto_weights": proto_weights,
            "proto_vectors": self.prototype_bank.prototypes.unsqueeze(0).expand(images.shape[0], -1, -1),
            "z_reason": z_reason,
            "disease_logits": disease_logits,
        }


__all__ = ["LesionDiseaseNet"]
