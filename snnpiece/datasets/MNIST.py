import torchvision
import torch
from .transforms import VecToLatencies

default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            VecToLatencies()
        ]
    )

def get_MNIST_dataloaders(data_path, batch_size,
                          transform=default_transform):

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader