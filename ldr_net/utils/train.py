import random

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, optimizer_cfg):
    backbone_params = []
    main_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            main_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": optimizer_cfg["lr_backbone"],
        },
        {
            "params": main_params,
            "lr": optimizer_cfg["lr_main"],
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=optimizer_cfg["weight_decay"])
