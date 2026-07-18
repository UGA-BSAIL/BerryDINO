from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import BlueberryFruitDataset
from losses_metrics import BCEDiceLoss, SegmentationMetrics
from runtime import build_model, count_parameters, get_device, set_seed


def make_loader(config, split, train):
    dataset = BlueberryFruitDataset(
        root=config.dataset_root,
        split=split,
        image_size=config.image_size,
        label_mode=config.label_mode,
        image_dir_name=config.image_dir_name,
        label_dir_name=config.label_dir_name,
        mask_dir_name=config.mask_dir_name,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size if train else 1,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    total_loss = 0.0

    for images, masks, _, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        metrics.update(logits, masks)

    result = metrics.compute()
    result["loss"] = total_loss / max(len(loader), 1)
    return result


def save_checkpoint(path, model, optimizer, epoch, best_miou, config, metrics):
    checkpoint = {
        "model_state": model.trainable_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "config": config.to_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def main():
    config = Config()
    config.validate()
    set_seed(config.seed)
    device = get_device()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    train_loader = make_loader(config, "train", train=True)
    val_loader = make_loader(config, "val", train=False)

    model = build_model(config, device)
    total_params, trainable_params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
    )
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    log_path = output_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_miou",
        "val_iou_fg",
        "val_iou_bg",
        "val_dice",
        "val_precision",
        "val_recall",
        "val_pixel_acc",
    ]

    best_miou = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for step, (images, masks, _, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_iou_fg": val_metrics["iou_fg"],
            "val_iou_bg": val_metrics["iou_bg"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_pixel_acc": val_metrics["pixel_acc"],
        }

        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            epoch,
            best_miou,
            config,
            val_metrics,
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(
                output_dir / "best_miou.pth",
                model,
                optimizer,
                epoch,
                best_miou,
                config,
                val_metrics,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss={train_loss:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"mIoU={val_metrics['miou']:.4f} | "
            f"Dice={val_metrics['dice']:.4f} | "
            f"P={val_metrics['precision']:.4f} | "
            f"R={val_metrics['recall']:.4f}"
        )

    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {output_dir / 'best_miou.pth'}")


if __name__ == "__main__":
    main()
