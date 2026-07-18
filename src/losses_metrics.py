from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6))
        return self.bce_weight * bce + self.dice_weight * dice_loss.mean()


@dataclass
class SegmentationMetrics:
    threshold: float = 0.5

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        predictions = torch.sigmoid(logits) >= self.threshold
        targets = targets >= 0.5

        self.tp += torch.logical_and(predictions, targets).sum().item()
        self.fp += torch.logical_and(predictions, torch.logical_not(targets)).sum().item()
        self.fn += torch.logical_and(torch.logical_not(predictions), targets).sum().item()
        self.tn += torch.logical_and(
            torch.logical_not(predictions), torch.logical_not(targets)
        ).sum().item()

    def compute(self):
        eps = 1e-7
        iou_fg = self.tp / (self.tp + self.fp + self.fn + eps)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + eps)
        miou = 0.5 * (iou_fg + iou_bg)
        dice = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        pixel_acc = (self.tp + self.tn) / (
            self.tp + self.tn + self.fp + self.fn + eps
        )
        return {
            "miou": float(miou),
            "iou_fg": float(iou_fg),
            "iou_bg": float(iou_bg),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "pixel_acc": float(pixel_acc),
        }
