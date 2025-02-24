import os

import torch.nn as nn
from torchvision import transforms

ROOT_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../dataset/",
)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
