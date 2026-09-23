"""
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

REFACTORED: For running unlearning methods with advanced MIA metrics.
All parameters are configurable via command-line arguments.

This script is similar to main.py but uses forget_new_metrics.py which includes
LiRA, Quantile, and Shadow MIA attacks.

Usage examples:
    # ResNet18 + Cifar10 with baseline
    python main_new_metrics.py -net ResNet18 -dataset Cifar10 -method baseline -seed 1 \
        -classes 10 -forget_per_class 500 -csv_path splits/split_indices_Cifar10.csv \
        -weight_path "Base Models/ResNet18-Cifar10-20-best.pth" -gpu
    
    # With retrain percentage (retrain25)
    python main_new_metrics.py -net ResNet18 -dataset Cifar10 -method retrain -ret_perc 25 \
        -seed 1 -classes 10 -forget_per_class 500 \
        -csv_path splits/split_indices_Cifar10.csv \
        -weight_path "Base Models/ResNet18-Cifar10-20-best.pth" -gpu
"""

import random
import os
import sys
import argparse
import time
from datetime import datetime
from typing import Tuple, List
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.optim as optim

import models
import datasets
import conf
import forget_new_metrics


def transfer_percent_from_forget_to_retain(trainset, forget_indices, retain_indices, 
                                            percent_to_transfer, num_classes=10, dataset_name="Cifar10"):
    """
    Moves the first X% (per class) of forget_indices into retain_indices.
    Returns updated forget_indices, retain_indices.
    """
    if dataset_name == 'MUCAC':
        targets = np.array(trainset.labels)
    else:
        targets = np.array(trainset.targets)
    updated_forget = []
    updated_retain = list(retain_indices)

    forget_by_class = {c: [] for c in range(num_classes)}
    for idx in forget_indices:
        label = targets[idx]
        forget_by_class[label].append(idx)

    for class_label in range(num_classes):
        class_forget = forget_by_class[class_label]
        class_forget = sorted(class_forget)

        num_to_transfer = int(len(class_forget) * percent_to_transfer)
        to_transfer = class_forget[:num_to_transfer]
        to_keep = class_forget[num_to_transfer:]

        updated_retain.extend(to_transfer)
        updated_forget.extend(to_keep)

    updated_forget_labels = [targets[i] for i in updated_forget]
    updated_retain_labels = [targets[i] for i in updated_retain]

    print("\nUpdated class distribution:")
    print("Retain set:")
    for c in range(num_classes):
        print(f"  Class {c}: {updated_retain_labels.count(c)}")
    print("Forget set:")
    for c in range(num_classes):
        print(f"  Class {c}: {updated_forget_labels.count(c)}")

    return updated_forget, updated_retain


def load_forget_retain_indices(trainset, dataset_name, seed, forget_per_class, 
                                csv_path, num_classes):
    """
    Load forget and retain indices for a given seed from CSV.
    Returns: forget_indices, retain_indices
    """
    df = pd.read_csv(csv_path)
    df_seed = df[
        (df['seed'] == seed) &
        (df['dataset'] == dataset_name) &
        (df['forget_per_class'] == forget_per_class)
    ]

    forget_indices = df_seed[df_seed['split'] == 'forget']['index'].tolist()
    retain_indices = df_seed[df_seed['split'] == 'retain']['index'].tolist()
    
    if dataset_name == 'MUCAC':
        forget_labels = [trainset.labels[i] for i in forget_indices]
        retain_labels = [trainset.labels[i] for i in retain_indices]
    else:
        forget_labels = [trainset.targets[i] for i in forget_indices]
        retain_labels = [trainset.targets[i] for i in retain_indices]

    print(f"\nRetain class distribution for seed {seed}:")
    if dataset_name == "Mnist":
        retain_dist = Counter([label.item() for label in retain_labels])
    else: 
        retain_dist = Counter(retain_labels)
    for c in range(num_classes): 
        print(f"Class {c}: {retain_dist.get(c, 0)}")

    print(f"\nForget class distribution for seed {seed}:")
    if dataset_name == "Mnist":
        forget_dist = Counter([label.item() for label in forget_labels])
    else: 
        forget_dist = Counter(forget_labels)
    for c in range(num_classes): 
        print(f"Class {c}: {forget_dist.get(c, 0)}")
        
    return forget_indices, retain_indices


def get_data_root(dataset_name):
    """Get the data root for a dataset."""
    if dataset_name == "PinsFaceRecognition":
        return "105_classes_pins_dataset"
    elif dataset_name == "MUCAC":
        return "./data/MUCAC"
    return "./data"


def save_results(results_dict, output_path):
    """Save results to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run unlearning methods with advanced MIA metrics")
    
    # Model and dataset
    parser.add_argument("-net", type=str, default="ResNet18", 
                        help="Network architecture: ResNet18, AllCNN, ViT")
    parser.add_argument("-dataset", type=str, default="Cifar10",
                        choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition", "Mnist", "MUCAC"],
                        help="Dataset to use")
    parser.add_argument("-classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("-weight_path", type=str, required=True,
                        help="Path to model weights")
    
    # Split configuration
    parser.add_argument("-csv_path", type=str, required=True,
                        help="Path to split indices CSV file")
    parser.add_argument("-forget_per_class", type=int, required=True,
                        help="Number of samples to forget per class")
    
    # Method configuration
    parser.add_argument("-method", type=str, default="baseline",
                        choices=["baseline", "retrain", "finetune", "teacher", 
                                 "amnesiac", "FisherForgetting", "ssdtuning"],
                        help="Unlearning method to run")
    parser.add_argument("-seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("-ret_perc", type=int, default=0,
                        help="Percentage from forget set to move to retain (0, 25, 50, 75)")
    
    # Training parameters
    parser.add_argument("-b", type=int, default=256,
                        help="Batch size")
    parser.add_argument("-gpu", action="store_true", default=False,
                        help="Use GPU")
    parser.add_argument("-lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("-warm", type=int, default=1,
                        help="Warm up epochs")
    parser.add_argument("-epochs", type=int, default=1,
                        help="Number of epochs for unlearning method")
    
    # Retrain model loading
    parser.add_argument("-retrain_folder", type=str, default=None,
                        help="Path to folder containing pre-trained retrain models")
    
    # Output
    parser.add_argument("-output_dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("-output_file", type=str, default=None,
                        help="Output filename (optional)")
    
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    batch_size = args.b

    print("=" * 60)
    print(f"Running: {args.net} + {args.dataset}")
    print(f"Method: {args.method} | Seed: {args.seed} | ret_perc: {args.ret_perc}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    net = getattr(models, args.net)(num_classes=args.classes)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    print(f"Loaded model from: {args.weight_path}")

    # Create unlearning teacher
    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

    if device == "cuda":
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    # Get data root and image size
    root = get_data_root(args.dataset)
    img_size = 224 if args.net == "ViT" else 32
    if args.dataset == "MUCAC":
        img_size = 128

    # Load full datasets
    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True, img_size=img_size
    )
    validset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=True, img_size=img_size
    )

    trainloader = DataLoader(trainset, num_workers=4, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, num_workers=4, batch_size=batch_size, shuffle=False)

    # Load forget/retain indices
    forget_indices, retain_indices = load_forget_retain_indices(
        trainset=trainset,
        dataset_name=args.dataset,
        seed=args.seed,
        forget_per_class=args.forget_per_class,
        csv_path=args.csv_path,
        num_classes=args.classes
    )

    # Transfer percentage if specified
    if args.ret_perc > 0:
        forget_indices, retain_indices = transfer_percent_from_forget_to_retain(
            trainset=trainset,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            percent_to_transfer=(args.ret_perc / 100),
            num_classes=args.classes,
            dataset_name=args.dataset,
        )

    # Create retain and forget sets with training transforms (unlearning=False)
    retainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=False,
        img_size=img_size, indices=retain_indices
    )
    forgetset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=False,
        img_size=img_size, indices=forget_indices
    )

    # Create retain and forget sets with evaluation transforms (unlearning=True)
    retainset_eval = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True,
        img_size=img_size, indices=retain_indices
    )
    forgetset_eval = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True,
        img_size=img_size, indices=forget_indices
    )

    # Training dataloaders
    retain_train_dl = DataLoader(retainset, batch_size=batch_size, shuffle=True, num_workers=4)
    forget_train_dl = DataLoader(forgetset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Evaluation dataloaders (original transforms for MIA)
    retain_train_dl_original = DataLoader(retainset_eval, batch_size=batch_size, shuffle=True, num_workers=4)
    forget_train_dl_original = DataLoader(forgetset_eval, batch_size=batch_size, shuffle=False, num_workers=4)

    # Use training dataloaders for validation as well
    forget_valid_dl = forget_train_dl
    retain_valid_dl = retain_train_dl
    forget_valid_dl_original = forget_train_dl_original
    retain_valid_dl_original = retain_train_dl_original

    # Full training dataloader
    full_train_dl = DataLoader(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        batch_size=batch_size,
    )

    # Model size scaler for SSD
    model_size_scaler = 1 if args.net != "ViT" else 0.5

    # Build kwargs for unlearning method
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
        # Original dataloaders for advanced MIA
        "retain_train_dl_original": retain_train_dl_original,
        "retain_valid_dl_original": retain_valid_dl_original,
        "forget_train_dl_original": forget_train_dl_original,
        "forget_valid_dl_original": forget_valid_dl_original,
        "dampening_constant": 1,
        "selection_weighting": 10 * model_size_scaler,
        "num_classes": args.classes,
        "dataset_name": args.dataset,
        "device": device,
        "model_name": args.net,
        "ret_perc": args.ret_perc,
        "lr": args.lr,
        "warm": args.warm,
    }

    # Add retrain_folder if specified
    if args.retrain_folder:
        kwargs["retrain_folder"] = args.retrain_folder

    # Start time tracking
    start = time.time()

    # Perform the unlearning method
    print(f"\nRunning {args.method} method...")
    testacc, retainacc, zrf, mia, mia_forget_retain, mia_forget_test, mia_retain_test, mia_train_test, d_f = getattr(
        forget_new_metrics, args.method
    )(**kwargs)

    # End time tracking
    elapsed = time.time() - start

    # Print the results
    print(f"\nTest Accuracy: {testacc}")
    print(f"Retain Accuracy: {retainacc}")
    print(f"Zero-Retain Forget (ZRF): {zrf}")
    print(f"Membership Inference Attack (MIA): {mia}")
    print(f"Forget vs Retain Membership Inference Attack (MIA): {mia_forget_retain}")
    print(f"Forget vs Test Membership Inference Attack (MIA): {mia_forget_test}")
    print(f"Test vs Retain Membership Inference Attack (MIA): {mia_retain_test}")
    print(f"Train vs Test Membership Inference Attack (MIA): {mia_train_test}")
    print(f"Forget Set Accuracy (Df): {d_f}")
    print(f"Method Execution Time: {elapsed:.2f} seconds")

    # Save results if output specified
    if args.output_dir:
        # Determine unlearning name based on ret_perc
        if args.ret_perc > 0:
            unlearning_name = f"{args.method}{args.ret_perc}"
        else:
            unlearning_name = args.method
        
        results_dict = {
            "unlearning": unlearning_name,
            "dataset": args.dataset.lower(),
            "model": args.net.lower(),
            "seed": args.seed,
            "Test Accuracy": testacc,
            "Retain Accuracy": retainacc,
            "Zero-Retain Forget (ZRF)": zrf,
            "Membership Inference Attack (MIA)": mia,
            "Forget vs Retain Membership Inference Attack (MIA)": mia_forget_retain,
            "Forget vs Test Membership Inference Attack (MIA)": mia_forget_test,
            "Test vs Retain Membership Inference Attack (MIA)": mia_retain_test,
            "Train vs Test Membership Inference Attack (MIA)": mia_train_test,
            "Forget Set Accuracy (Df)": d_f,
            "Method Execution Time": elapsed,
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.output_file:
            output_path = os.path.join(args.output_dir, args.output_file)
        else:
            output_path = os.path.join(
                args.output_dir, 
                f"{unlearning_name}_{args.dataset.lower()}_{args.net.lower()}_seed_{args.seed}.txt"
            )
        
        save_results(results_dict, output_path)


if __name__ == '__main__':
    main()
