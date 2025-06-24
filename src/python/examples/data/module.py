import torch
import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10(pl.LightningDataModule):
    def __init__(self, batch_size=64, transform=None):
        super().__init__()
        self.cifar10_val = None
        self.cifar10_train = None
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self):
        # Downloads only
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_train = datasets.CIFAR10(root='./data', train=True, transform=self.transform)
        self.cifar10_val = datasets.CIFAR10(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=2)


def resnet18transform(weights=None) -> transforms.Compose:
    if weights:
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def resnet50transform(weights=None) -> transforms.Compose:
    if weights:
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def resnet101transform(weights=None) -> transforms.Compose:
    if weights:
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def resnet152transform(weights=None) -> transforms.Compose:
    if weights:
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
