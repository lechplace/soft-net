"""
BackboneRegistry — catalog of SOTA pretrained models available via Keras Applications.

Each backbone entry defines:
  - factory: callable (weights, include_top, input_shape) → keras.Model
  - default_input_size: recommended (H, W) for the backbone
  - family: architecture family (efficientnet, resnet, vgg, ...)
  - min_input_size: smallest supported (H, W)

Usage:
    from softnet.image.backbones import BackboneRegistry
    backbone = BackboneRegistry.get("efficientnet_b0", weights="imagenet")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    default_input_size: tuple[int, int]
    min_input_size: tuple[int, int]
    factory: Callable


def _lazy(module_path: str, class_name: str) -> Callable:
    """Lazily import Keras Application to avoid loading all of TF at import time."""
    def _factory(weights="imagenet", include_top=False, input_shape=None):
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(weights=weights, include_top=include_top, input_shape=input_shape)
    return _factory


_BACKBONES: dict[str, BackboneSpec] = {
    # ── EfficientNet family ────────────────────────────────────────────
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        family="efficientnet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "EfficientNetB0"),
    ),
    "efficientnet_b3": BackboneSpec(
        name="efficientnet_b3",
        family="efficientnet",
        default_input_size=(300, 300),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "EfficientNetB3"),
    ),
    "efficientnet_b7": BackboneSpec(
        name="efficientnet_b7",
        family="efficientnet",
        default_input_size=(600, 600),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "EfficientNetB7"),
    ),
    "efficientnetv2_s": BackboneSpec(
        name="efficientnetv2_s",
        family="efficientnetv2",
        default_input_size=(384, 384),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "EfficientNetV2S"),
    ),
    "efficientnetv2_l": BackboneSpec(
        name="efficientnetv2_l",
        family="efficientnetv2",
        default_input_size=(480, 480),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "EfficientNetV2L"),
    ),
    # ── ResNet family ──────────────────────────────────────────────────
    "resnet50": BackboneSpec(
        name="resnet50",
        family="resnet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "ResNet50"),
    ),
    "resnet50v2": BackboneSpec(
        name="resnet50v2",
        family="resnet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "ResNet50V2"),
    ),
    "resnet101v2": BackboneSpec(
        name="resnet101v2",
        family="resnet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "ResNet101V2"),
    ),
    # ── MobileNet family ───────────────────────────────────────────────
    "mobilenet_v2": BackboneSpec(
        name="mobilenet_v2",
        family="mobilenet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "MobileNetV2"),
    ),
    "mobilenet_v3_small": BackboneSpec(
        name="mobilenet_v3_small",
        family="mobilenet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "MobileNetV3Small"),
    ),
    "mobilenet_v3_large": BackboneSpec(
        name="mobilenet_v3_large",
        family="mobilenet",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "MobileNetV3Large"),
    ),
    # ── ConvNeXt family ────────────────────────────────────────────────
    "convnext_tiny": BackboneSpec(
        name="convnext_tiny",
        family="convnext",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "ConvNeXtTiny"),
    ),
    "convnext_base": BackboneSpec(
        name="convnext_base",
        family="convnext",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "ConvNeXtBase"),
    ),
    # ── Xception ──────────────────────────────────────────────────────
    "xception": BackboneSpec(
        name="xception",
        family="xception",
        default_input_size=(299, 299),
        min_input_size=(71, 71),
        factory=_lazy("keras.applications", "Xception"),
    ),
    # ── VGG (legacy reference) ─────────────────────────────────────────
    "vgg16": BackboneSpec(
        name="vgg16",
        family="vgg",
        default_input_size=(224, 224),
        min_input_size=(32, 32),
        factory=_lazy("keras.applications", "VGG16"),
    ),
}

# Alias for recommended default
_BACKBONES["default"] = _BACKBONES["efficientnet_b0"]


class BackboneRegistry:
    """Access and inspect available pretrained backbones."""

    @classmethod
    def list(cls, family: str | None = None) -> list[str]:
        names = [k for k in _BACKBONES if k != "default"]
        if family:
            names = [k for k in names if _BACKBONES[k].family == family]
        return sorted(names)

    @classmethod
    def get_spec(cls, name: str) -> BackboneSpec:
        if name not in _BACKBONES:
            available = cls.list()
            raise ValueError(
                f"Unknown backbone '{name}'. Available: {available}"
            )
        return _BACKBONES[name]

    @classmethod
    def get(cls, name: str, weights: str = "imagenet") -> object:
        """Instantiate backbone (feature extractor, no top)."""
        spec = cls.get_spec(name)
        h, w = spec.default_input_size
        return spec.factory(weights=weights, include_top=False, input_shape=(h, w, 3))

    @classmethod
    def families(cls) -> list[str]:
        return sorted({v.family for k, v in _BACKBONES.items() if k != "default"})
