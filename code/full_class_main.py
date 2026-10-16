#!/bin/python3.8
"""
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707
"""


import random
import os

# import optuna
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
import forget_full
import datasets
import models
import conf
from training_utils import *


"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", default="AllCNN", type=str,  help="net type")
    
    # Set default path relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
default_weight_path = os.path.join(current_dir, "checkpoint", "AllCNN", "AllCNN-Cifar20-40-best.pth")

parser.add_argument(
        "-weight_path",
        type=str,
        default=default_weight_path,
        help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
        "-dataset",
        type=str,
        default = "Cifar20",
        nargs="?",
        choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition", "Mnist", "MUCAC"],
        help="dataset to train on",
)
parser.add_argument(
        "-ret_perc", type=int, default=0, help="percentage from forget set to move to retrain"
    )    
parser.add_argument("-classes", default=20,  type=int, required=False, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    required=False,
    nargs="?",
    default = "baseline",
    choices=[
        "baseline",
        "retrain",
        "finetune",
        "teacher",
        "amnesiac",
        "UNSIR",
        "NTK",
        "ssdtuning",
        "FisherForgetting",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_class",
    type=str,
    default = "electrical_devices",
    required=False,
    nargs="?",
    help="class to forget",
    choices=list(conf.class_dict),
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
parser.add_argument(
    "-retrain_folder",
    type=str,
    default=None,
    help="Path to folder containing pre-trained retrain models (avoids retraining from scratch)",
)
args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# Check that the correct things were loaded
if args.dataset == "Cifar20":
    assert args.forget_class in conf.cifar20_classes
elif args.dataset == "Cifar100":
    assert args.forget_class in conf.cifar100_classes

print(conf.class_dict)
forget_class = conf.class_dict[args.forget_class]

print(forget_class)

batch_size = args.b


# get network
net = getattr(models, args.net)(num_classes=args.classes)
net.load_state_dict(torch.load(args.weight_path))

# for bad teacher
unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

if args.gpu:
    net = net.cuda()
    unlearning_teacher = unlearning_teacher.cuda()

# For celebritiy faces
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

# Scale for ViT (faster training, better performance)
img_size = 224 if args.net == "ViT" else 32
trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=True, img_size=img_size
)
validset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=True, img_size=img_size
)

# Set up the dataloaders and prepare the datasets
trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)

classwise_train, classwise_test = forget_full_class_strategies.get_classwise_ds(
    trainset, args.classes
), forget_full_class_strategies.get_classwise_ds(validset, args.classes)

(
    retain_train,
    retain_valid,
    forget_train,
    forget_valid,
) = forget_full_class_strategies.build_retain_forget_sets(
    classwise_train, classwise_test, args.classes, forget_class, args.ret_perc
)


retainset = getattr(datasets, args.dataset)(
        root=root,
        download=True,
        train=True,
        unlearning=False,
        img_size=img_size,
        indices=retain_train
    )
forgetset = getattr(datasets, args.dataset)(
        root=root,
        download=True,
        train=True,
        unlearning=False,
        img_size=img_size,
        indices=forget_train
    )

forget_train_dl = DataLoader(forgetset, batch_size)
retain_train_dl = DataLoader(retainset, batch_size, shuffle=True)




full_train_dl = DataLoader(
    ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    batch_size=batch_size,
)

forget_valid_dl = forget_train_dl   #DataLoader(forget_valid, batch_size)
retain_valid_dl = retain_train_dl  #DataLoader(retain_valid, batch_size)



# Change alpha here as described in the paper
# For PinsFaceRe-cognition, we use α=50 and λ=0.1
model_size_scaler = 1
if args.net == "ViT":
    model_size_scaler = 0.5
else:
    model_size_scaler = 1

kwargs = {
        "model": net,
        "seed": args.seed,
        "unlearning_teacher": unlearning_teacher,
        "train_dl": trainloader,
        "retain_train_dl": retain_train_dl,
        "retain_valid_dl": retain_valid_dl,
        "forget_train_dl": forget_train_dl,
        "forget_valid_dl": forget_valid_dl,
        "full_train_dl": full_train_dl,
        "valid_dl": validloader,
        "dampening_constant": 1,
        "selection_weighting": 10 * model_size_scaler,
        "num_classes": args.classes,
        "dataset_name": args.dataset,
        "device": "cuda" if args.gpu else "cpu",
        "model_name": args.net,
      "ret_perc": args.ret_perc,
      "retrain_folder": args.retrain_folder,
}


# Time the method
import time

start = time.time()

# executes the method passed via args
testacc, retainacc, zrf, mia, mia_forget_retain, mia_forget_test, mia_retain_test, mia_train_test, d_f = getattr(forget_full, args.method)( 
        **kwargs
    )

# End time tracking
end = time.time()
time_elapsed = end - start
    
# Print the results
print(f"Test Accuracy: {testacc}")
print(f"Retain Accuracy: {retainacc}")
print(f"Zero-Retain Forget (ZRF): {zrf}")
print(f"Membership Inference Attack (MIA): {mia}")
print(f"Forget vs Retain Membership Inference Attack (MIA): {mia_forget_retain}")
print(f"Forget vs Test Membership Inference Attack (MIA): {mia_forget_test}")
print(f"Test vs Retain Membership Inference Attack (MIA): {mia_retain_test}")
print(f"Train vs Test Membership Inference Attack (MIA): {mia_train_test}")
print(f"Forget Set Accuracy (Df): {d_f}")
print(f"Method Execution Time: {time_elapsed:.2f} seconds")
    
