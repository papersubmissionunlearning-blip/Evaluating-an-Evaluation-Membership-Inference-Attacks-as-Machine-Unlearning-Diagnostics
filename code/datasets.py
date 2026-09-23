"""
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

FIXED: 
- Added use_augmentation parameter to control augmentation independently
- Fixed list mutation bug (now copies lists before appending)
- Fixed MUCAC split inversion
"""

from typing import Any, Tuple
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import os
import glob
from PIL import Image

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Cropping etc. to improve performance of the model (details see https://github.com/weiaicunzai/pytorch-cifar100)
# NOTE: These are now templates - ALWAYS copy before modifying!
_transform_train_from_scratch = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

_transform_unlearning = [
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

_transform_test = [
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]


def get_cifar_transform(train: bool, unlearning: bool, use_augmentation: bool, img_size: int):
    """
    Get the appropriate transform for CIFAR datasets.
    
    Args:
        train: Whether this is training data
        unlearning: Whether this is for unlearning evaluation (no augmentation)
        use_augmentation: Whether to use augmentation (can override train behavior)
        img_size: Image size for resize transform
    
    Returns:
        Composed transform
    """
    if train:
        if unlearning:
            # Evaluation mode - no augmentation
            transform = list(_transform_test)
        elif use_augmentation:
            # Training with augmentation
            transform = list(_transform_train_from_scratch)
        else:
            # Training WITHOUT augmentation (overfitted case)
            transform = list(_transform_test)
    else:
        # Test set - never augmented
        transform = list(_transform_test)
    
    transform.append(transforms.Resize(img_size))
    return transforms.Compose(transform)


class Cifar100(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, 
                 indices=None, use_augmentation=True):
        transform = get_cifar_transform(train, unlearning, use_augmentation, img_size)
        super().__init__(root=root, train=train, download=download, transform=transform)
        
        # Store parameters for reconstruction
        self.img_size = img_size
        self.unlearning = unlearning
        self.use_augmentation = use_augmentation
        
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar20(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, 
                 indices=None, use_augmentation=True):
        transform = get_cifar_transform(train, unlearning, use_augmentation, img_size)
        super().__init__(root=root, train=train, download=download, transform=transform)
        
        # Store parameters for reconstruction
        self.img_size = img_size
        self.unlearning = unlearning
        self.use_augmentation = use_augmentation

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y
    


class Cifar10(CIFAR10):
    def __init__(self, root, train, unlearning, download, img_size=32, 
                 indices=None, use_augmentation=True):
        transform = get_cifar_transform(train, unlearning, use_augmentation, img_size)
        super().__init__(root=root, train=train, download=download, transform=transform)
        
        # Store parameters for reconstruction
        self.img_size = img_size
        self.unlearning = unlearning
        self.use_augmentation = use_augmentation

        # Subsample data and targets if indices are given
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

# MNIST transform templates - ALWAYS copy before modifying!
_transform_mnist_train = [
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
]

_transform_mnist_test = [
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
]


def get_mnist_transform(train: bool, unlearning: bool, use_augmentation: bool, img_size: int):
    """Get the appropriate transform for MNIST dataset."""
    if train:
        if unlearning:
            transform = list(_transform_mnist_test)
        elif use_augmentation:
            transform = list(_transform_mnist_train)
        else:
            transform = list(_transform_mnist_test)
    else:
        transform = list(_transform_mnist_test)
    
    # Insert grayscale conversion at the beginning
    transform.insert(0, transforms.Grayscale(num_output_channels=3))
    transform.append(transforms.Resize(img_size))
    return transforms.Compose(transform)


class Mnist(MNIST):
    def __init__(self, root, train, unlearning, download, img_size=32, 
                 indices=None, use_augmentation=True):
        transform = get_mnist_transform(train, unlearning, use_augmentation, img_size)
        super().__init__(root=root, train=train, download=download, transform=transform)
        
        # Store parameters for reconstruction
        self.img_size = img_size
        self.unlearning = unlearning
        self.use_augmentation = use_augmentation
        
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y
    

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y


# MUCAC recommended transforms
def get_mucac_transform(train: bool, unlearning: bool, use_augmentation: bool):
    """Get the appropriate transform for MUCAC dataset."""
    if train and not unlearning and use_augmentation:
        return transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(128), 
            transforms.ToTensor(),
        ])


# Identity and label loading (run once per script)
def load_mucac_meta(path):
    # Load identities
    with open(os.path.join(path, "CelebA-HQ-identity.txt")) as f:
        identities = dict(line.strip().split() for line in f.readlines())
    
    # Load attributes
    attributes_map = {"smiling": 32}
    label_map = {}
    with open(os.path.join(path, "CelebA-HQ-attribute.txt")) as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.strip().split()
            file_name = parts[0]
            label_map[file_name] = {attr: int(parts[idx]) for attr, idx in attributes_map.items()}
    return identities, label_map


class MUCAC(Dataset):
    """
    MUCAC Dataset with FIXED split logic.
    
    FIXED: The original code had inverted split semantics where:
    - train=True, unlearning=False loaded the FORGET identity range
    - train=True, unlearning=True loaded the TRAIN identity range
    
    Now correctly:
    - train=True, unlearning=False: loads TRAIN identity range (190-1969) for training
    - train=True, unlearning=True: loads FORGET identity range (1970-4854) for forget set
    - train=False: loads TEST identity range (0-189)
    """
    def __init__(self, root, train, unlearning, download=False, img_size=128, 
                 indices=None, use_augmentation=True):
        self.root = root
        self.train = train
        self.unlearning = unlearning
        self.img_size = img_size
        self.use_augmentation = use_augmentation

        # Identity ranges (FIXED semantics)
        self.test_max_identity = 190           # 0-189: test set
        self.train_max_identity = 1970         # 190-1969: train (retain) set  
        self.forget_max_identity = 4855        # 1970-4854: forget set
        
        self.img_dir = os.path.join(self.root, "CelebAMask-HQ", "CelebA-HQ-img")
        
        print(f"MUCAC img_dir: {self.img_dir}")
        
        # Load metadata
        self.identities, self.label_map = load_mucac_meta(self.root)

        # Transforms
        self.transform = get_mucac_transform(train, unlearning, use_augmentation)

        # Select files based on identity range - FIXED LOGIC
        self.image_paths = []
        self.labels = []

        for img_path in glob.glob(os.path.join(self.img_dir, "*.jpg")):
            file_name = os.path.basename(img_path)
            identity = int(self.identities[file_name])
            smiling = self.label_map[file_name]["smiling"]
            if smiling == -1:
                smiling = 0

            # FIXED: Correct split logic
            if train:
                if unlearning:
                    # Forget set: identities 1970-4854
                    if self.train_max_identity <= identity < self.forget_max_identity:
                        self.image_paths.append(img_path)
                        self.labels.append(smiling)
                else:
                    # Train (retain) set: identities 190-1969
                    if self.test_max_identity <= identity < self.train_max_identity:
                        self.image_paths.append(img_path)
                        self.labels.append(smiling)
            else:
                # Test set: identities 0-189
                if identity < self.test_max_identity:
                    self.image_paths.append(img_path)
                    self.labels.append(smiling)

        print(f"MUCAC split - train={train}, unlearning={unlearning}: {len(self.image_paths)} samples")
        
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, torch.Tensor([]), label
