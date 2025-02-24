import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .constant import ROOT_DATA_DIR, TRANSFORM


class ImgDataset(Dataset):
    def __init__(
        self,
        dataset,
    ) -> None:
        self.dataset = dataset
        self.class_to_idx = {cls: idx for idx,
                             cls in enumerate(dataset.classes)}
        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


def get_dataloader(
    batch_size: int = 32,
    dataset_path: str = ROOT_DATA_DIR,
    transform: transforms.Compose = TRANSFORM,
):
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform,
    )
    dataset = ImgDataset(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
