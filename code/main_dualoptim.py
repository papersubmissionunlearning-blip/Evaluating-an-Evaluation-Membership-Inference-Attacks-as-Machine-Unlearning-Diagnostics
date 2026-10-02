"""Experiment runner for the six DualOptim unlearning methods.

A copy of `main.py`'s `main()` with three changes:
  1. `-method` accepts scrub / salun / sfron / scrub_do / salun_do / sfron_do
  2. those names dispatch into `forget_dualoptim` instead of `forget`
  3. optional flags can override the paper hyperparameters

The split, dataset and loader construction below is byte-for-byte the same as
`main.py`, and the four split/IO helpers are *imported* from `main.py` rather
than copied, so the forget/retain partition for a given seed is guaranteed
identical to the one used by the existing nine methods. That identity is what
makes the new CSV rows comparable to the existing ones.

Defaults follow Table 7 of the DualOptim paper for CIFAR-10 / ResNet-18 with a
10% random forget set. Batch size is split in two: `-b` (default 128, the
paper's value) sizes the loaders the unlearning algorithms iterate, while
`-eval_b` (default 256, matching `main.py`) sizes the loaders the metrics are
computed on. They are separate because `utils.validation_epoch_end` averages
accuracy across batches rather than across samples, so a short final batch is
over-weighted and the reported accuracy shifts with batch size. Keeping
evaluation at 256 leaves the accuracy columns comparable to the other nine
methods without changing the optimization.

The MIA columns are unaffected by either setting: `metrics.collect_prob`
rebuilds its loader from `.dataset` at a fixed batch size of 128 with
shuffle=False, so it ignores the batching of whatever loader it is handed.

Usage:
    python main_dualoptim.py -net ResNet18 -dataset Cifar10 -method scrub_do \
        -seed 1 -classes 10 -forget_per_class 500 \
        -csv_path splits/split_indices_Cifar10.csv \
        -weight_path "Base Models/ResNet18-Cifar10-20-best.pth" \
        -b 128 -gpu -output_dir "Results_Fixed/Results_ResNet18_Cifar10"
"""

import random
import os
import sys
import argparse
import time
import json
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
import forget
import forget_dualoptim
import conf
from unlearn import *
from utils import *
from training_utils import *

# Single-sourced from main.py so the forget/retain split cannot drift
from main import (
    transfer_percent_from_forget_to_retain,
    load_forget_retain_indices,
    get_data_root,
    save_results,
)

DUALOPTIM_METHODS = forget_dualoptim.METHODS


def resolve_method(method_name):
    """Return the unlearning function, preferring the DualOptim implementations."""
    if method_name in DUALOPTIM_METHODS:
        return getattr(forget_dualoptim, method_name)
    return getattr(forget, method_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run DualOptim unlearning experiments"
    )

    # Model and dataset
    parser.add_argument("-net", type=str, default="ResNet18",
                        help="Network architecture: ResNet18, AllCNN, ViT")
    parser.add_argument("-dataset", type=str, default="Cifar10",
                        choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition", "Mnist", "MUCAC"],
                        help="Dataset to use")
    parser.add_argument("-classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("-weight_path", type=str, required=True,
                        help="Path to baseline model weights")

    # Split configuration
    parser.add_argument("-csv_path", type=str, required=True,
                        help="Path to split indices CSV file")
    parser.add_argument("-forget_per_class", type=int, required=True,
                        help="Number of samples to forget per class")

    # Method configuration
    parser.add_argument("-method", type=str, default="scrub",
                        choices=["baseline", "retrain", "finetune", "teacher",
                                 "amnesiac", "FisherForgetting", "ssdtuning"]
                                + DUALOPTIM_METHODS,
                        help="Unlearning method to use")
    parser.add_argument("-seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("-ret_perc", type=int, default=0,
                        help="Percentage from forget set to move to retain")

    # Training configuration
    # NOTE: default 128 (paper batch size), unlike main.py's 256.
    # This applies to the unlearning loaders only -- see -eval_b.
    parser.add_argument("-b", type=int, default=128,
                        help="Batch size for the unlearning loaders")
    # Evaluation batch size is deliberately decoupled from -b and defaults to
    # 256, matching main.py. `validation_epoch_end` averages accuracy over
    # batches rather than over samples, so a short final batch is over-weighted
    # and the reported accuracy depends on the batch size. Pinning evaluation to
    # 256 keeps the accuracy columns directly comparable to the other methods
    # while unlearning still runs at the paper's 128.
    parser.add_argument("-eval_b", type=int, default=256,
                        help="Batch size for the evaluation loaders (metrics)")
    parser.add_argument("-lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("-warm", type=int, default=1,
                        help="Warm up epochs")
    parser.add_argument("-gpu", action="store_true", default=False,
                        help="Use GPU")

    # Augmentation control (for overfitted experiments)
    parser.add_argument("-no_augmentation", action="store_true", default=False,
                        help="Disable data augmentation (for overfitted experiments)")

    # Epochs override (for underfitted experiments)
    parser.add_argument("-epochs_override", type=int, default=None,
                        help="Override default epochs (for underfitted experiments)")
    parser.add_argument("-milestones_override", type=str, default=None,
                        help="Override milestones as comma-separated values, e.g., '8,12,16'")

    # Retrain folder (to load pre-existing retrain models)
    parser.add_argument("-retrain_folder", type=str, default=None,
                        help="Path to folder containing pre-trained retrain models (avoids retraining)")

    # DualOptim hyperparameter overrides. Left unset, the paper's Table 7
    # values for CIFAR-10 / ResNet-18 are used.
    parser.add_argument("-optim", type=str, default=None,
                        choices=["sgd", "adam", "dual", "dual_as", "dual_aa"],
                        help="Override the optimizer configuration")
    parser.add_argument("-unlearn_lr", type=float, default=None,
                        help="Override the forget learning rate (eta_f)")
    parser.add_argument("-retain_lr", type=float, default=None,
                        help="Override the retain learning rate (eta_r)")
    parser.add_argument("-unlearn_epochs", type=int, default=None,
                        help="Override unlearning epochs (SCRUB/SalUn)")
    parser.add_argument("-sfron_iters", type=int, default=None,
                        help="Override SFRon outer iterations (T_out)")
    parser.add_argument("-sfron_freq", type=int, default=None,
                        help="Override SFRon inner steps (T_in)")
    parser.add_argument("-sfron_alpha", type=float, default=None,
                        help="Override SFRon alpha")

    # Output
    parser.add_argument("-output_dir", type=str, default=None,
                        help="Directory to save results (optional)")
    parser.add_argument("-output_file", type=str, default=None,
                        help="Output filename (optional, auto-generated if not provided)")

    args = parser.parse_args()

    # Parse milestones if provided
    milestones_override = None
    if args.milestones_override:
        milestones_override = [int(x) for x in args.milestones_override.split(',')]

    # Collect only the hyperparameters the user explicitly overrode
    dualoptim_overrides = {
        key: getattr(args, key)
        for key in ("optim", "unlearn_lr", "retain_lr", "unlearn_epochs",
                    "sfron_iters", "sfron_freq", "sfron_alpha")
        if getattr(args, key) is not None
    }

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Determine augmentation setting
    use_augmentation = not args.no_augmentation

    print("=" * 60)
    print(f"Running: {args.net} + {args.dataset}")
    print(f"Method: {args.method} | Seed: {args.seed}")
    print(f"Classes: {args.classes} | Forget per class: {args.forget_per_class}")
    print(f"Augmentation: {use_augmentation}")
    print(f"Batch size: {args.b} (unlearning) | {args.eval_b} (evaluation)")
    if dualoptim_overrides:
        print(f"DualOptim overrides: {dualoptim_overrides}")
    if args.epochs_override:
        print(f"Epochs override: {args.epochs_override}")
    print("=" * 60)

    batch_size = args.b            # unlearning loaders (paper value)
    eval_batch_size = args.eval_b  # metric loaders (matches the other methods)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Load model
    net = getattr(models, args.net)(num_classes=args.classes)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    print(f"Loaded model from: {args.weight_path}")

    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

    if device == "cuda":
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    # Get data root and image size
    root = get_data_root(args.dataset)
    img_size = 224 if args.net == "ViT" else 32
    if args.dataset == "MUCAC":
        img_size = 128

    # Load datasets with configurable augmentation
    # MUCAC: Use "train_all" to load full pool (190-4854) for consistent indexing with split file
    mucac_identity_range = "train_all" if args.dataset == "MUCAC" else None

    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True,
        img_size=img_size, use_augmentation=use_augmentation,
        identity_range=mucac_identity_range
    )
    validset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=True,
        img_size=img_size, use_augmentation=use_augmentation
    )

    # Evaluation-only loaders: valid_dl feeds Test Accuracy, train_dl feeds
    # mia_train_test. Both use eval_batch_size.
    trainloader = DataLoader(trainset, num_workers=4, batch_size=eval_batch_size, shuffle=True)
    validloader = DataLoader(validset, num_workers=4, batch_size=eval_batch_size, shuffle=False)

    # Load forget/retain indices
    # Cifar20 uses fine CIFAR-100 labels (0-99) in trainset.targets, not coarse labels (0-19)
    num_classes_for_split = 100 if args.dataset == 'Cifar20' else args.classes

    forget_indices, retain_indices = load_forget_retain_indices(
        trainset=trainset,
        dataset_name=args.dataset,
        seed=args.seed,
        forget_per_class=args.forget_per_class,
        csv_path=args.csv_path,
        num_classes=num_classes_for_split
    )

    # Transfer percentage if specified
    if args.ret_perc > 0:
        forget_indices, retain_indices = transfer_percent_from_forget_to_retain(
            trainset=trainset,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            percent_to_transfer=(args.ret_perc / 100),
            num_classes=num_classes_for_split,
            dataset_name=args.dataset,
        )

    # Create retain and forget sets (with augmentation for training)
    # MUCAC: Use "train_all" to load full pool (190-4854) for consistent indexing
    retainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=False,
        img_size=img_size, indices=retain_indices, use_augmentation=use_augmentation,
        identity_range=mucac_identity_range
    )
    forgetset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=False,
        img_size=img_size, indices=forget_indices, use_augmentation=use_augmentation,
        identity_range=mucac_identity_range
    )

    # Evaluation sets (no augmentation)
    # MUCAC: Use "train_all" for consistent indexing with split file
    retainset_eval = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True,
        img_size=img_size, indices=retain_indices, use_augmentation=use_augmentation,
        identity_range=mucac_identity_range
    )
    forgetset_eval = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True,
        img_size=img_size, indices=forget_indices, use_augmentation=use_augmentation,
        identity_range=mucac_identity_range
    )

    retain_train_dl = DataLoader(retainset, batch_size=batch_size, shuffle=True)
    forget_train_dl = DataLoader(forgetset, batch_size=batch_size, shuffle=False)

    forget_valid_dl = forget_train_dl
    retain_valid_dl = retain_train_dl

    # These feed Retain Accuracy, Df accuracy and ZRF, so they use
    # eval_batch_size rather than the unlearning batch size.
    retain_train_dl_original = DataLoader(retainset_eval, batch_size=eval_batch_size, shuffle=True)
    forget_train_dl_original = DataLoader(forgetset_eval, batch_size=eval_batch_size, shuffle=False)

    forget_valid_dl_original = forget_train_dl_original
    retain_valid_dl_original = retain_train_dl_original

    # Model size scaler for SSD
    model_size_scaler = 0.5 if args.net == "ViT" else 1

    full_train_dl = DataLoader(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        batch_size=batch_size,
    )

    # Build kwargs for the method
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
        "dualoptim_overrides": dualoptim_overrides or None,
    }

    # Add epochs override if specified
    if args.epochs_override is not None:
        kwargs["epochs_override"] = args.epochs_override
        kwargs["milestones_override"] = milestones_override

    # Add retrain folder if specified (for loading pre-existing retrain models)
    if args.retrain_folder:
        kwargs["retrain_folder"] = args.retrain_folder

    # Start time tracking
    start = time.time()

    # Perform the unlearning method
    results = resolve_method(args.method)(**kwargs)
    testacc, retainacc, zrf, mia, mia_forget_retain, mia_forget_test, mia_retain_test, mia_train_test, d_f = results

    # End time tracking
    elapsed = time.time() - start

    # Convert tensors to floats
    if torch.is_tensor(zrf):
        zrf = float(zrf)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {testacc}")
    print(f"Retain Accuracy: {retainacc}")
    print(f"Zero-Retain Forget (ZRF): {zrf}")
    print(f"Membership Inference Attack (MIA): {mia}")
    print(f"Forget vs Retain MIA: {mia_forget_retain}")
    print(f"Forget vs Test MIA: {mia_forget_test}")
    print(f"Test vs Retain MIA: {mia_retain_test}")
    print(f"Train vs Test MIA: {mia_train_test}")
    print(f"Forget Set Accuracy (Df): {d_f}")
    print(f"Method Execution Time: {elapsed:.2f} seconds")

    # Save results if output specified
    if args.output_dir:
        # Determine unlearning name based on ret_perc
        # ret_perc=0 -> method name as-is (e.g., "scrub")
        # ret_perc>0 -> method name + ret_perc (e.g., "scrub25")
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
