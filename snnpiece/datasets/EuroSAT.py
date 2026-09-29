import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'

def random_split(dataset, ratio=0.8, random_state=None):
    """
    Randomly split a dataset into two non-overlapping subsets.

    Args:
    - dataset (Dataset): dataset to be split
    - ratio (float): ratio of the size of the first subset. Defaults to 0.8
    - random_state (int): seed for random number generator. Defaults to None

    Returns:
    - A tuple of two datasets, the first containing int(ratio * len(dataset)) elements,
      and the second containing the rest.
    """
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split

class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        """
        Args:
        root (str): root directory of the dataset
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        """
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)

class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1
    

default_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(size=(16,16)),
                    transforms.ToTensor(),
                ]
            )

def get_EuroSAT_dataloaders(data_path, batch_size, seed=42424242,
                          transform=default_transform):

    dataset = EuroSAT(
            root=data_path,
            transform=transform,
        )
    train_ds, test_ds = random_split(dataset, 0.8, random_state=seed)

    # load train dataset with computed normalization
    train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        
    # load val dataset
    test_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, test_loader

