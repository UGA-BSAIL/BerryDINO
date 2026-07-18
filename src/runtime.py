from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from model import AgriDINOSeg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config, device):
    backbone = torch.hub.load(
        config.dino_repo,
        config.model_name,
        source="local",
        weights=config.dino_weights,
    )
    model = AgriDINOSeg(
        backbone=backbone,
        patch_size=config.patch_size,
        intermediate_layers=config.intermediate_layers,
        decoder_channels=config.decoder_channels,
        num_classes=config.num_classes,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    return model


def load_trainable_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    trainable_missing = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in missing:
            trainable_missing.append(name)

    if trainable_missing:
        raise RuntimeError(
            "Checkpoint is missing trainable parameters:\n" + "\n".join(trainable_missing)
        )
    if unexpected:
        raise RuntimeError("Unexpected checkpoint keys:\n" + "\n".join(unexpected))
    return checkpoint


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
