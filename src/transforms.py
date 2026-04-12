"""
Image augmentation and preprocessing pipelines.

Three pipelines are provided:
  get_train_transforms(config)  — augmentation + normalization (training)
  get_val_transforms(config)    — deterministic resize + normalization (val/test)
  get_inference_transforms()    — same as val; used in webcam and ONNX export

All pipelines normalize with ImageNet mean/std so pretrained ResNet weights
remain valid.
"""

from __future__ import annotations

from torchvision import transforms

from src.config import Config, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(config: Config) -> transforms.Compose:
    """Return the augmented training transform pipeline."""
    aug_list = [
        transforms.Resize((config.image_size, config.image_size)),
    ]

    if config.use_horizontal_flip:
        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if config.use_brightness_jitter:
        aug_list.append(
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            )
        )

    if config.use_blur:
        aug_list.append(
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
        )

    aug_list += [
        transforms.RandomGrayscale(p=0.02),          # minor robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(aug_list)


def get_val_transforms(config: Config) -> transforms.Compose:
    """Return the deterministic validation / test transform pipeline."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic inference pipeline (no config dependency).

    Suitable for use in infer_webcam.py and export_onnx.py where a full
    Config object may not be available.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
