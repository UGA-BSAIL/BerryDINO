from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _index_by_stem(folder: Path, extensions=None) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate stem '{path.stem}' in {folder}")
        result[path.stem] = path
    return result


def yolo_polygons_to_mask(label_path: Path, width: int, height: int) -> Image.Image:
    """Rasterize all YOLO-seg polygons into one binary semantic mask."""
    mask = Image.new("L", (width, height), 0)
    if not label_path.exists():
        return mask

    draw = ImageDraw.Draw(mask)
    with label_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                coords = [float(v) for v in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Bad YOLO polygon in {label_path}:{line_num}") from exc

            if len(coords) % 2 != 0:
                continue

            points = []
            for x, y in zip(coords[0::2], coords[1::2]):
                px = min(max(x * width, 0.0), width - 1.0)
                py = min(max(y * height, 0.0), height - 1.0)
                points.append((px, py))

            if len(points) >= 3:
                draw.polygon(points, fill=255)

    return mask


def letterbox_pair(
    image: Image.Image,
    mask: Image.Image,
    target_size: int,
) -> Tuple[Image.Image, Image.Image, dict]:
    """Resize with unchanged aspect ratio and pad to a square."""
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image_resized = TF.resize(
        image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR
    )
    mask_resized = TF.resize(
        mask, [new_height, new_width], interpolation=InterpolationMode.NEAREST
    )

    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left
    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top

    image_padded = TF.pad(
        image_resized,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=0,
    )
    mask_padded = TF.pad(
        mask_resized,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=0,
    )

    meta = {
        "original_width": width,
        "original_height": height,
        "resized_width": new_width,
        "resized_height": new_height,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }
    return image_padded, mask_padded, meta


class JointTransform:
    def __init__(self, image_size: int, train: bool):
        self.image_size = image_size
        self.train = train
        self.color_jitter = ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() < 0.35:
                angle = random.uniform(-15.0, 15.0)
                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )

            if random.random() < 0.5:
                image = self.color_jitter(image)

        image, mask, meta = letterbox_pair(image, mask, self.image_size)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, meta


class BlueberryFruitDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        label_mode: str = "yolo",
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        mask_dir_name: str = "masks",
        train: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.label_mode = label_mode
        self.transform = JointTransform(image_size=image_size, train=train)

        split_dir = self.root / split
        image_map = _index_by_stem(split_dir / image_dir_name, IMAGE_EXTENSIONS)

        if label_mode == "yolo":
            target_map = _index_by_stem(split_dir / label_dir_name, {".txt"})
        elif label_mode == "mask":
            target_map = _index_by_stem(split_dir / mask_dir_name, IMAGE_EXTENSIONS)
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        missing = sorted(set(image_map) - set(target_map))
        extra = sorted(set(target_map) - set(image_map))
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(f"{split}: {len(missing)} images have no target. Examples: {preview}")
        if extra:
            preview = ", ".join(extra[:10])
            raise RuntimeError(f"{split}: {len(extra)} targets have no image. Examples: {preview}")

        self.samples: List[Tuple[str, Path, Path]] = [
            (stem, image_map[stem], target_map[stem]) for stem in sorted(image_map)
        ]
        if not self.samples:
            raise RuntimeError(f"No samples found in split: {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem, image_path, target_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.label_mode == "yolo":
            mask = yolo_polygons_to_mask(target_path, *image.size)
        else:
            mask_np = np.asarray(Image.open(target_path).convert("L"))
            mask = Image.fromarray((mask_np > 0).astype(np.uint8) * 255, mode="L")

        image_tensor, mask_tensor, meta = self.transform(image, mask)
        return image_tensor, mask_tensor, stem, meta
