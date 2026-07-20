# BerryDINO

BerryDINO is a DINOv3-based semantic segmentation framework developed for blueberry vision applications, spanning harvest-stage fruit segmentation and postharvest bruise segmentation.

BerryDINO uses a frozen DINOv3 backbone together with three task-specific modules:

- Adaptive Layer-Scale Aggregation (ALSA);
- Bidirectional Hierarchical Feature Decoder (BHD);
- Multi-Receptive Gated Refinement (MGR).

<p align="center">
  <img src="assets/BerryDINO_Overall_Architecture.jpg" width="800">
  <br>
  <em><strong>Fig.1</strong> Overall architecture of BerryDINO.</em>
</p>

## Repository Structure

```text
BerryDINO/
├── checkpoints/
│   ├── bruise_seg/
│   └── fruit_seg/
├── data/
│   ├── README.md
│   └── dataset_sample.jpg
├── dinov3-main/
│   └── README.md
├── dinov3_weights/
│   └── README.md
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── losses_metrics.py
│   ├── model.py
│   ├── runtime.py
│   ├── train.py
│   └── test.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup Instructions

### 1. Clone BerryDINO

```bash
git clone https://github.com/UGA-BSAIL/BerryDINO.git
cd BerryDINO
```

### 2. Create a Python environment

```bash
conda create -n berrydino python=3.10
conda activate berrydino
pip install -r requirements.txt
```

A PyTorch installation compatible with the available CUDA version and GPU is recommended. On HiPerGator, users should select the appropriate CUDA and PyTorch environment for the target GPU partition.

### 3. Prepare DINOv3

The official DINOv3 source code and pretrained backbone weights are not redistributed in this repository.

Download the official DINOv3 source code and place its contents in:

```text
dinov3-main/
```

Download the required pretrained DINOv3 backbone weights and place them in:

```text
dinov3_weights/
```

The prepared directories should follow this general structure:

```text
dinov3-main/
├── dinov3/
├── hubconf.py
└── ...

dinov3_weights/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── ...
```

See the following files for additional instructions:

- `dinov3-main/README.md`
- `dinov3_weights/README.md`

Additional dependencies required by DINOv3 should be installed according to the official DINOv3 repository.

## Data Information

The Blueberry Semantic Segmentation Dataset (Berry-Seg) supports blueberry fruit segmentation, blueberry bruise segmentation, and independent external validation.

The dataset is publicly available on Kaggle:

**[Download the Berry-Seg dataset from Kaggle](KAGGLE_DATASET_LINK)**(Coming Soon...)

Dataset statistics, representative samples, and the expected directory structure are provided in:

```text
data/README.md
```

After downloading the dataset, set the corresponding dataset path in:

```text
src/config.py
```

## Pretrained Checkpoints

Pretrained BerryDINO (From S to L) checkpoints are provided for the two primary segmentation tasks:

```text
checkpoints/
├── bruise_seg/
│   └── ViT-S
│       └── best_miou.pth
│   └── ViT-S+
│       └── best_miou.pth
│   └── ViT-B
│       └── best_miou.pth
│   └── ViT-L
│       └── best_miou.pth
└── fruit_seg/
│   └── ViT-S
│       └── best_miou.pth
│   └── ViT-S+
│       └── best_miou.pth
│   └── ViT-B
│       └── best_miou.pth
│   └── ViT-L
        └── best_miou.pth
```

These checkpoints contain the trainable BerryDINO parameters. The frozen DINOv3 ViT-S/16 backbone weights must be downloaded separately.

## Configuration

Before training or evaluation, configure the paths and model settings in:

```text
src/config.py
```

The main settings include:

- dataset path;
- DINOv3 source-code path;
- DINOv3 pretrained-weight path;
- checkpoint path;
- output directory;
- DINOv3 backbone;
- intermediate feature layers;
- training hyperparameters.

The selected backbone, weight file, and intermediate-layer indices must correspond to the same DINOv3 model.

## Usage

Run all commands from the BerryDINO repository root.

### Training

```bash
python src/train.py
```

The training script saves the best validation checkpoint, the latest checkpoint, training logs, and experiment configuration.

### Testing

```bash
python src/test.py
```

The testing script generates global metrics, per-image metrics, predicted masks, and prediction overlays.

## Default Training Settings

| Setting | Value |
|---|---|
| Input size | 512 × 512 |
| Batch size | 4 |
| Epochs | 20 |
| Learning rate | 1 × 10⁻⁴ |
| Optimizer | AdamW |
| Loss | BCE + Dice loss |

BerryDINO supports the following DINOv3 backbones:

- ViT-S/16;
- ViT-S+/16;
- ViT-B/16;
- ViT-L/16.

## Citation

The BerryDINO manuscript and formal citation information will be added after publication.

```bibtex
@article{berrydino,
  title   = {BerryDINO: ...},
  author  = {Wang, Rui-Feng and ... and Li, Changying},
  journal = {...},
  year    = {...}
}
```

## License

This project is released under the license specified in the `LICENSE` file.

The DINOv3 source code and pretrained weights remain subject to their original licenses and terms of use.
