from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class MultiReceptiveGatedRefinement(nn.Module):
    """
    Lightweight refinement with:
      - local depthwise 3x3 branch
      - dilated depthwise 3x3 branch
      - channel gate
      - zero-initialized residual scale
    """
    def __init__(self, channels: int, gate_reduction: int = 4):
        super().__init__()
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.context = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_valid_group_count(channels), channels)
        self.act = nn.GELU()

        hidden = max(channels // gate_reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.local(x) + self.context(x)
        refined = self.act(self.norm(self.mix(refined)))
        refined = refined * self.gate(x)
        return x + self.gamma * refined


class AdaptiveLayerScaleAdapter(nn.Module):
    """
    Every pyramid scale receives information from all selected DINO layers.
    Softmax-normalized weights learn the layer contribution at each scale.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.projections = nn.ModuleList(
            [ConvNormAct(in_channels, out_channels, kernel_size=1) for _ in range(num_layers)]
        )

        # Four target scales: 1/4, 1/8, 1/16, 1/32.
        self.layer_weights = nn.Parameter(torch.zeros(4, num_layers))

        self.scale_post = nn.ModuleList(
            [MultiReceptiveGatedRefinement(out_channels) for _ in range(4)]
        )

    @staticmethod
    def _resize(feature: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        input_hw: Tuple[int, int],
    ) -> List[torch.Tensor]:
        if len(features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} features, received {len(features)}")

        projected = [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]

        input_h, input_w = input_hw
        target_sizes = [
            (input_h // 4, input_w // 4),
            (input_h // 8, input_w // 8),
            (input_h // 16, input_w // 16),
            (input_h // 32, input_w // 32),
        ]

        outputs: List[torch.Tensor] = []
        for scale_index, target_size in enumerate(target_sizes):
            weights = torch.softmax(self.layer_weights[scale_index], dim=0)
            fused = 0.0
            for layer_index, feature in enumerate(projected):
                resized = self._resize(feature, target_size)
                fused = fused + weights[layer_index] * resized
            outputs.append(self.scale_post[scale_index](fused))
        return outputs


class WeightedFusion(nn.Module):
    """Learned positive normalized weights for same-resolution feature fusion."""
    def __init__(self, num_inputs: int, channels: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.refine = MultiReceptiveGatedRefinement(channels)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        weights = F.softplus(self.weights)
        weights = weights / (weights.sum() + 1e-6)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        return self.refine(fused)


class BidirectionalHierarchicalDecoder(nn.Module):
    def __init__(self, channels: int, num_classes: int = 1):
        super().__init__()
        # Top-down
        self.top_p5 = MultiReceptiveGatedRefinement(channels)
        self.top_p4 = WeightedFusion(2, channels)
        self.top_p3 = WeightedFusion(2, channels)
        self.top_p2 = WeightedFusion(2, channels)

        # Bottom-up
        self.down_2_to_3 = ConvNormAct(channels, channels, kernel_size=3, stride=2, groups=channels)
        self.down_3_to_4 = ConvNormAct(channels, channels, kernel_size=3, stride=2, groups=channels)
        self.bottom_p3 = WeightedFusion(3, channels)
        self.bottom_p4 = WeightedFusion(3, channels)

        # Final aggregation at 1/4 resolution.
        self.aggregate = nn.Sequential(
            ConvNormAct(channels * 3, channels, kernel_size=1),
            MultiReceptiveGatedRefinement(channels),
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = pyramid  # 1/4, 1/8, 1/16, 1/32

        t5 = self.top_p5(p5)
        t4 = self.top_p4([p4, self._up_to(t5, p4)])
        t3 = self.top_p3([p3, self._up_to(t4, p3)])
        t2 = self.top_p2([p2, self._up_to(t3, p2)])

        b3 = self.bottom_p3([p3, t3, self.down_2_to_3(t2)])
        b4 = self.bottom_p4([p4, t4, self.down_3_to_4(b3)])

        merged = torch.cat(
            [
                t2,
                self._up_to(b3, t2),
                self._up_to(b4, t2),
            ],
            dim=1,
        )
        return self.classifier(self.aggregate(merged))


class AgriDINOSeg(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int = 16,
        intermediate_layers: Sequence[int] = (2, 5, 8, 11),
        decoder_channels: int = 128,
        num_classes: int = 1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.intermediate_layers = tuple(intermediate_layers)
        self.freeze_backbone_flag = freeze_backbone

        embed_dim = int(self.backbone.embed_dim)
        self.adapter = AdaptiveLayerScaleAdapter(
            in_channels=embed_dim,
            out_channels=decoder_channels,
            num_layers=len(self.intermediate_layers),
        )
        self.decoder = BidirectionalHierarchicalDecoder(
            channels=decoder_channels,
            num_classes=num_classes,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone_flag:
            self.backbone.eval()
        return self

    @staticmethod
    def _to_feature_map(
        feature,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        if isinstance(feature, (tuple, list)):
            feature = feature[0]

        if feature.ndim == 4:
            return feature

        if feature.ndim != 3:
            raise ValueError(f"Unexpected intermediate feature shape: {tuple(feature.shape)}")

        num_patches = patch_h * patch_w
        if feature.shape[1] < num_patches:
            raise ValueError(
                f"Feature token count {feature.shape[1]} is smaller than patch count {num_patches}"
            )
        if feature.shape[1] != num_patches:
            # DINOv3 sequence order: extra tokens first, patch tokens last.
            feature = feature[:, -num_patches:, :]

        return feature.transpose(1, 2).reshape(
            feature.shape[0], feature.shape[-1], patch_h, patch_w
        )

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        context = torch.no_grad() if self.freeze_backbone_flag else torch.enable_grad()
        with context:
            raw_features = self.backbone.get_intermediate_layers(
                x,
                n=list(self.intermediate_layers),
            )

        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size
        return [
            self._to_feature_map(feature, patch_h, patch_w)
            for feature in raw_features
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        features = self._extract_features(x)
        pyramid = self.adapter(features, input_hw=input_hw)
        logits = self.decoder(pyramid)
        return F.interpolate(
            logits,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )

    def trainable_state_dict(self):
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        state = self.state_dict()
        return {name: value for name, value in state.items() if name in trainable_names}
