# DINOv3 Pretrained Weights

The official **DINOv3** pretrained backbone weights are not included in this repository.

Please download the required pretrained weights from the official [***DINOv3***](https://github.com/facebookresearch/dinov3#overview) repository and place them in this directory.

## Supported DINOv3 Pretrained Weights

BerryDINO supports the following DINOv3 ViT backbones pretrained on the LVD-1689M web dataset.

| Model | Parameters | Pretraining Dataset |
| --- | ---: | --- |
| ViT-S/16 distilled | 21M | LVD-1689M |
| ViT-S+/16 distilled | 29M | LVD-1689M |
| ViT-B/16 distilled | 86M | LVD-1689M |
| ViT-L/16 distilled | 300M | LVD-1689M |

## Expected Weight Files

Place the downloaded pretrained weights directly in this directory using the following filenames:

```text
dinov3_weights/
├── README.md
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Please ensure that the selected `model_name`, pretrained weight file, and `intermediate_layers` setting in `src/config.py` correspond to the same DINOv3 backbone.
