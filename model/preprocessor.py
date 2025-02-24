from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision import datasets, transforms


def get_transform(
    image_size: List[int] = [224, 224],
    augmentation: bool = False,
) -> transforms.Compose:
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    if augmentation:
        transform_list.insert(
            1,
            transforms.RandomHorizontalFlip(),
        )

    return transforms.Compose(transform_list)


def preprocess_img(
    img_path: str,
    transforms: transforms.Compose,
):
    image = Image.open(img_path)
    image = image.convert("RGB")

    return transforms(image).unsqueeze(0)
