from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import BlueberryFruitDataset
from losses_metrics import SegmentationMetrics
from runtime import build_model, count_parameters, get_device, load_trainable_checkpoint


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    image = (image_tensor * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[..., 0] = 255
    alpha = 0.45
    selected = mask.astype(bool)
    overlay[selected] = (
        (1.0 - alpha) * overlay[selected] + alpha * tint[selected]
    ).astype(np.uint8)
    return overlay


@torch.no_grad()
def main():
    config = Config()
    config.validate()
    device = get_device()

    checkpoint_path = Path(config.output_dir) / "best_miou.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dataset = BlueberryFruitDataset(
        root=config.dataset_root,
        split="test",
        image_size=config.image_size,
        label_mode=config.label_mode,
        image_dir_name=config.image_dir_name,
        label_dir_name=config.label_dir_name,
        mask_dir_name=config.mask_dir_name,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = build_model(config, device)
    checkpoint = load_trainable_checkpoint(model, checkpoint_path, device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print(f"Loaded epoch: {checkpoint.get('epoch')}")

    output_root = Path(config.output_dir) / "test_results"
    mask_dir = output_root / "pred_masks"
    overlay_dir = output_root / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    metrics = SegmentationMetrics(threshold=config.threshold)
    rows = []

    for images, masks, stems, _ in tqdm(test_loader, desc="Testing"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        metrics.update(logits, masks)

        probs = torch.sigmoid(logits)
        preds = probs >= config.threshold

        # Per-image confusion and metrics.
        one = SegmentationMetrics(threshold=config.threshold)
        one.update(logits, masks)
        sample_metrics = one.compute()
        sample_metrics["id"] = stems[0]
        rows.append(sample_metrics)

        pred_array = preds[0, 0].cpu().numpy().astype(np.uint8)
        Image.fromarray(pred_array * 255, mode="L").save(
            mask_dir / f"{stems[0]}.png"
        )

        image_array = denormalize_image(images[0])
        overlay = make_overlay(image_array, pred_array)
        Image.fromarray(overlay, mode="RGB").save(
            overlay_dir / f"{stems[0]}.png"
        )

    global_metrics = metrics.compute()
    output_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "miou",
        "iou_fg",
        "iou_bg",
        "dice",
        "precision",
        "recall",
        "pixel_acc",
    ]
    with (output_root / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "global_metrics.json").write_text(
        json.dumps(global_metrics, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    for key, value in global_metrics.items():
        print(f"{key:>12s}: {value:.4f}")
    print("=" * 60)
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
