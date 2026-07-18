from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    # ---------------- Paths ----------------
    dataset_root: str = "./datasets/blueberry_fruit_seg"

    # dataset_root: str = "./datasets/blueberry_bruise_seg"

    dino_repo: str = "./dinov3-main"
    dino_weights: str = (
        "./dinov3_weights/"
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        # "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
        # "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    )
    output_dir: str = "./out/runs/blueberry_fruit_dinov3s"
    # output_dir: str = "./out/runs/blueberry_fruit_dinov3splus"
    # output_dir: str = "./out/runs/blueberry_fruit_dinov3b"
    # output_dir: str = "./out/runs/blueberry_fruit_dinov3l"

    # output_dir: str = "./out/runs/blueberry_bruise_dinov3s"
    # output_dir: str = "./out/runs/blueberry_bruise_dinov3splus"
    # output_dir: str = "./out/runs/blueberry_bruise_dinov3b"
    # output_dir: str = "./out/runs/blueberry_bruise_dinov3l"

    # Expected dataset structure:
    # dataset_root/
    #   train/images/*.{jpg,png,...}
    #   train/labels/*.txt       # YOLO segmentation polygons
    #   val/images/*
    #   val/labels/*.txt
    #   test/images/*
    #   test/labels/*.txt
    #
    # PNG masks are also supported:
    #   train/masks/<same_stem>.png
    # Set label_mode="mask" in that case.
    label_mode: str = "yolo"  # "yolo" or "mask"
    image_dir_name: str = "images"
    label_dir_name: str = "labels"
    mask_dir_name: str = "masks"

    # ---------------- Model ----------------
    model_name: str = "dinov3_vits16"
    # model_name: str = "dinov3_vits16plus"
    # model_name: str = "dinov3_vitb16"
    # model_name: str = "dinov3_vitl16"
    patch_size: int = 16
    intermediate_layers: tuple = (2, 5, 8, 11)
    # intermediate_layers: tuple = (5, 11, 17, 23)
    decoder_channels: int = 128
    num_classes: int = 1
    freeze_backbone: bool = True

    # ---------------- Training ----------------
    image_size: int = 512  # must be divisible by patch_size
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    threshold: float = 0.5
    amp: bool = True
    bce_weight: float = 1.0
    dice_weight: float = 1.0

    def validate(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.label_mode not in {"yolo", "mask"}:
            raise ValueError("label_mode must be 'yolo' or 'mask'")
        for path_name in ("dataset_root", "dino_repo", "dino_weights"):
            p = Path(getattr(self, path_name))
            if not p.exists():
                raise FileNotFoundError(f"{path_name} does not exist: {p}")

    def to_dict(self):
        data = asdict(self)
        data["intermediate_layers"] = list(self.intermediate_layers)
        return data
