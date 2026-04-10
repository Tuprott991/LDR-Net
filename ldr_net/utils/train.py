import random

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _group_named_parameters(model):
    backbone_params = []
    main_params = []

    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            main_params.append(param)

    return backbone_params, main_params


def build_optimizer(model, optimizer_cfg):
    backbone_params, main_params = _group_named_parameters(model)

    param_groups = [
        {
            "params": backbone_params,
            "lr": optimizer_cfg["lr_backbone"],
            "group_name": "backbone",
        },
        {
            "params": main_params,
            "lr": optimizer_cfg["lr_main"],
            "group_name": "main",
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=optimizer_cfg["weight_decay"])


def set_backbone_trainable(model, trainable: bool):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
    model._freeze_backbone_active = not trainable


def configure_optimizer_for_epoch(model, optimizer, optimizer_cfg, epoch: int):
    freeze_backbone_epochs = int(optimizer_cfg.get("freeze_backbone_epochs", 0))
    backbone_is_frozen = epoch < freeze_backbone_epochs
    set_backbone_trainable(model, trainable=not backbone_is_frozen)

    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        if group_name == "backbone":
            group["lr"] = 0.0 if backbone_is_frozen else float(optimizer_cfg["lr_backbone"])
        elif group_name == "main":
            group["lr"] = float(optimizer_cfg["lr_main"])

    return {
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "backbone_frozen": backbone_is_frozen,
        "lr_backbone": 0.0 if backbone_is_frozen else float(optimizer_cfg["lr_backbone"]),
        "lr_main": float(optimizer_cfg["lr_main"]),
    }
