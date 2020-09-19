import torch
import logging
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.transforms import mnist_transforms, cifar_transforms

logger = logging.getLogger("train_log")

def create_datasets(dataset: str, path: str, channels: int=1):
    """Get datasets

    Args:
        dataset (str): Pass either MNIST or CIFAR10
        path (str): Path to download the datasets
    """
    if dataset == "MNIST":
        transform = mnist_transforms(crop_size=20)
        train = MNIST(root=path, train=True, download=True, transform=transform["train"])
        test = MNIST(root=path, train=False, download=True, transform=transform["test"])
    elif dataset == "CIFAR10":
        transform = cifar_transforms(augment=True, channels=channels)
        train = CIFAR10(root=path, train=True, download=True, transform=transform["train"])
        test = CIFAR10(root=path, train=False, download=True, transform=transform["test"])
    else:
        raise Exception("Dataset name is invalid.")
    
    logger.info("==> Dataset: {}".format(dataset))
    logger.info("==> Single training instance shape: {}".format(train[0][0].size()))
    # logger.info("==> Test data shape: {}, [max/min]: [{:.2f}/{:.2f}]".format(test[0][0].size(), test[0][0].max(), test[0][0].min()))
    return train, test

def create_dataloaders(datasets: tuple, batch_size: int=32, valid_pct: float=0.2, num_workers: int=4):
    """Get Dataloaders"""
    # obtain training indices that will be used for validation
    train, test = datasets
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_pct * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # load training data in batches
    trainloader = DataLoader(train,
                            batch_size=batch_size,
                            sampler=train_sampler,
                            num_workers=num_workers,
                            pin_memory=True
                            )
    # load validation data in batches
    validloader = DataLoader(train,
                            batch_size=batch_size,
                            sampler=valid_sampler,
                            num_workers=num_workers,
                            pin_memory=True
                            )
    # load test data in batches
    testloader = DataLoader(test,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True
                            )
    return trainloader, testloader, validloader
