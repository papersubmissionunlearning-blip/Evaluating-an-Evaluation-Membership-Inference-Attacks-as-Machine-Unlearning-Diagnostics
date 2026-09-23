"""
Train a model from scratch.

Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

REFACTORED: Added parameters for underfitted/overfitted experiments.
- epochs_override: Override default epochs (for underfitted)
- no_augmentation: Disable data augmentation (for overfitted)
- output_path: Specify output path for saved model

Usage examples:
    # Standard training (same as original)
    python train_model.py -net ResNet18 -dataset Cifar10 -classes 10 -gpu
    
    # Underfitted (1 epoch)
    python train_model.py -net ResNet18 -dataset Cifar10 -classes 10 \
        -epochs_override 1 -output_path checkpoint/underfitted/model.pth -gpu
    
    # Overfitted (no augmentation)
    python train_model.py -net ResNet18 -dataset Cifar10 -classes 10 \
        -no_augmentation -output_path checkpoint/overfitted/model.pth -gpu
"""

import random
import os
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
from unlearn import *
from utils import *
import forget
import datasets
import models
import conf
from training_utils import *
import pandas as pd 
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset
import numpy as np
from collections import Counter
from torchvision.datasets import CIFAR10 
import torchvision.transforms as transforms

# Original code from https://github.com/weiaicunzai/pytorch-cifar100
"""
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707
"""

def train(epoch):
    start = time.time()
    epoch_loss = 0.0
    correct_train = 0
    total_train_samples = 0
    net.train()
    for batch_index, (images, _, labels) in enumerate(trainloader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        #
        batch_size_actual = images.size(0)
        epoch_loss += loss.item() * batch_size_actual
        total_train_samples += batch_size_actual

        _, preds = outputs.max(1)
        correct_train += preds.eq(labels).sum().item()
        #
        
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(trainloader.dataset)
        ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()
    #
    avg_train_loss = epoch_loss / total_train_samples
    train_acc = correct_train / total_train_samples
    finish = time.time()
    print("Epoch {} - Average Train Loss: {:.4f}, Train Accuracy: {:.4f}".format(epoch, avg_train_loss, train_acc))    
    #
    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start,
        )
    )
    print()

    return correct.float() / len(testloader.dataset)


# Argument parsing - ORIGINAL + NEW PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default="ResNet18", help="net type")
parser.add_argument("-dataset", type=str, default="Cifar10", help="dataset to train on")
parser.add_argument("-classes", type=int, default=10, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=256, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")

# NEW PARAMETERS for underfitted/overfitted experiments
parser.add_argument("-epochs_override", type=int, default=None,
                    help="Override default epochs (for underfitted experiments)")
parser.add_argument("-milestones_override", type=str, default=None,
                    help="Override milestones as comma-separated values (e.g., '8,12,16')")
parser.add_argument("-no_augmentation", action="store_true", default=False,
                    help="Disable data augmentation (for overfitted experiments)")
parser.add_argument("-output_path", type=str, default=None,
                    help="Path to save the best model (overrides default checkpoint path)")
parser.add_argument("-seed", type=int, default=None,
                    help="Random seed for reproducibility")

args = parser.parse_args()

# Set seed if provided
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# Get milestones - use override if provided, else use config
if args.milestones_override:
    MILESTONES = [int(x) for x in args.milestones_override.split(',')]
else:
    MILESTONES = (
        getattr(conf, f"{args.dataset}_MILESTONES")
        if args.net != "ViT"
        else getattr(conf, f"{args.dataset}_ViT_MILESTONES")
    )

# Get epochs - use override if provided, else use config
if args.epochs_override is not None:
    EPOCHS = args.epochs_override
else:
    EPOCHS = (
        getattr(conf, f"{args.dataset}_EPOCHS")
        if args.net != "ViT"
        else getattr(conf, f"{args.dataset}_ViT_EPOCHS")
    )

# Determine whether to use augmentation
use_augmentation = not args.no_augmentation

print("=" * 60)
print(f"Training: {args.net} on {args.dataset}")
print(f"Classes: {args.classes}")
print(f"Epochs: {EPOCHS}")
print(f"Milestones: {MILESTONES}")
print(f"Augmentation: {use_augmentation}")
print(f"GPU: {args.gpu}")
print("=" * 60)

# Get network
net = getattr(models, args.net)(num_classes=args.classes)
if args.gpu:
    net = net.cuda()

# Dataloaders - ORIGINAL LOGIC PRESERVED
if args.dataset == "PinsFaceRecognition":
    root = "105_classes_pins_dataset"
elif args.dataset == "MUCAC":
    root = "./data/MUCAC"
else:
    root = "./data"

img_size = 224 if args.net == "ViT" else 128  # ORIGINAL: 128 for all except ViT

# Load datasets - pass use_augmentation parameter
trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=False, img_size=img_size,
    use_augmentation=use_augmentation
)
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size,
    use_augmentation=use_augmentation  # ORIGINAL: unlearning=False
)

trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
testloader = DataLoader(testset, batch_size=args.b, shuffle=False)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.2
)  # learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

# Checkpoint path - use output_path if provided, else use original logic
if args.output_path:
    # Create directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_path = args.output_path
    use_custom_path = True
else:
    checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, args.net, conf.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{dataset}-{epoch}-{type}.pth")
    use_custom_path = False

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)  # ORIGINAL: pass epoch

    train(epoch)
    acc = eval_training(epoch)

    if best_acc < acc:
        if use_custom_path:
            weights_path = checkpoint_path
        else:
            weights_path = checkpoint_path.format(
                net=args.net, dataset=args.dataset, epoch=epoch, type="best"
            )
        print("saving weights file to {}".format(weights_path))
        torch.save(net.state_dict(), weights_path)
        best_acc = acc
        continue

print("=" * 60)
print(f"Training complete. Best accuracy: {best_acc:.4f}")
print("=" * 60)
