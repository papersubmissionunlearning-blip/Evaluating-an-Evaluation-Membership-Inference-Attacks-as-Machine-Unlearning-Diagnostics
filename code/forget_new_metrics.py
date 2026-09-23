"""
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

FIXED:
- retrain() now loads pre-existing retrain models from disk if available
- retrain() now passes the correct model to get_metric_scores
"""
from torch.utils.data import Subset, DataLoader, ConcatDataset
import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
from tqdm import tqdm

from sklearn import linear_model, model_selection

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob, get_membership_attack_prob_our, evaluate_mia_xgboost
from utils import *
import ssd as ssd_
import conf
import models

import time
import os
import torch.nn as nn
import torch.optim as optim
from training_utils import WarmUpLR
from retrain_utils import find_best_retrain_model


def get_size_dl(name, dl):
    print(name, str(len(dl.dataset)))

def make_unlearning_loader_from_loader(loader, unlearning: bool = True):
    """
    Rebuild a DataLoader so that its (possibly custom) dataset is recreated with
    `unlearning=True`. Works with plain datasets and torch.utils.data.Subset.

    Assumes the underlying dataset's __init__ accepts kwargs like:
      root, download, train, img_size, indices, transform, target_transform, unlearning

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Existing loader to copy settings from.
    unlearning : bool
        Value to pass to the dataset's `unlearning` kwarg.

    Returns
    -------
    torch.utils.data.DataLoader
        New DataLoader whose dataset has unlearning=True.
    """
    ds = loader.dataset

    def _extract_common_kwargs(d):
        # Pull common constructor kwargs if present on the dataset instance
        kw = {}
        for k in ["root", "download", "train", "img_size", "indices",
                  "transform", "target_transform", "classes", "split"]:
            if hasattr(d, k):
                kw[k] = getattr(d, k)
        kw["unlearning"] = unlearning
        return kw

    # Rebuild dataset (handles Subset by rebuilding its base dataset, then re-wrapping)
    if isinstance(ds, Subset):
        base = ds.dataset
        base_cls = base.__class__
        base_kwargs = _extract_common_kwargs(base)
        try:
            new_base = base_cls(**base_kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to reconstruct base dataset with kwargs {base_kwargs}") from e
        new_ds = Subset(new_base, ds.indices)
    else:
        cls = ds.__class__
        kwargs = _extract_common_kwargs(ds)
        try:
            new_ds = cls(**kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to reconstruct dataset with kwargs {kwargs}") from e

    # Infer shuffle from sampler type
    def _infer_shuffle(dl):
        return isinstance(dl.sampler, torch.utils.data.RandomSampler)

    # Rebuild DataLoader, copying important settings
    new_loader = DataLoader(
        new_ds,
        batch_size=loader.batch_size,
        shuffle=_infer_shuffle(loader),
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
        collate_fn=getattr(loader, "collate_fn", None),
        persistent_workers=getattr(loader, "persistent_workers", False),
        prefetch_factor=getattr(loader, "prefetch_factor", None),
        generator=getattr(loader, "generator", None),
    )
    return new_loader
  
def get_metric_scores(
    model,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
    retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  device,
  dataset_name,
  model_name
):
    get_size_dl("Valid (Test) Dl: ", valid_dl)
    get_size_dl("Train Dl: ", train_dl)
    get_size_dl("Retain Train Dl: ", retain_train_dl)
    get_size_dl("Forget Train Dl: ", forget_train_dl)
    get_size_dl("Retain Valid Dl: ", retain_valid_dl)
    get_size_dl("Forget Valid Dl: ", forget_valid_dl)

    loss_acc_dict = evaluate(model, valid_dl, device)
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    d_f = evaluate(model, forget_valid_dl, device)
    #mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model) 
    mia = 0
    
    mia_forget_retain = get_membership_attack_prob_our(retain_train_dl, forget_train_dl, model)
    mia_forget_test = get_membership_attack_prob_our(valid_dl, forget_train_dl, model)
    mia_retain_test = get_membership_attack_prob_our(retain_train_dl, valid_dl, model)
    mia_train_test = get_membership_attack_prob_our(valid_dl, train_dl, model)
    
    #mia_forget_retain = evaluate_mia_xgboost(retain_train_dl_original, forget_train_dl_original, model)
    #mia_forget_test = evaluate_mia_xgboost(valid_dl, forget_train_dl_original, model)
    #mia_retain_test = evaluate_mia_xgboost(retain_train_dl_original, valid_dl, model)
    #mia_train_test = evaluate_mia_xgboost(valid_dl, train_dl, model)

    #mia_forget_retain = get_mia_lira(retain_train_dl, forget_train_dl, model, device, f"forget_retain_{dataset_name}_{model_name}")
    #mia_forget_test = get_mia_lira(valid_dl, forget_train_dl, model, device, f"forget_test_{dataset_name}_{model_name}")
    #mia_retain_test = get_mia_lira(retain_train_dl, valid_dl, model, device, f"retain_test_{dataset_name}_{model_name}")
    #mia_train_test = get_mia_quantile(train_dl, valid_dl, model, device, f"train_test_{dataset_name}_{model_name}")
    
    # loss_acc_dict =  0
    # retain_acc_dict = 0
    # zrf = 0
    # d_f = 0
    # mia = 0
    

    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, mia_forget_retain, mia_forget_test, mia_retain_test, mia_train_test, d_f["Acc"]) 


def baseline(
    model,
    seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
    retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
  dataset_name,
  model_name,
    device,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
      
      retain_train_dl_original,
      retain_valid_dl_original,
      forget_train_dl_original,
      forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )

def retrain(
    model,
    seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
    retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    # Config
    num_classes = kwargs.get("num_classes", 20)
    retrain_folder = kwargs.get("retrain_folder", None)
    
    # Initialize model
    net = getattr(models, model_name)(num_classes=num_classes)
    if device == "cuda" and torch.cuda.is_available():
        net = net.cuda()
    
    # ========== TRY TO LOAD EXISTING RETRAIN MODEL ==========
    existing_model_path = find_best_retrain_model(
        retrain_folder, model_name, dataset_name, seed,
        ret_perc=kwargs.get("ret_perc", 0),
    )
    
    if existing_model_path and os.path.exists(existing_model_path):
        print("=" * 60)
        print(f"✅ Loading existing retrain model: {existing_model_path}")
        print("=" * 60)
        net.load_state_dict(torch.load(existing_model_path, map_location=device))
        net.eval()
    else:
        # ========== TRAIN FROM SCRATCH ==========
        print("=" * 60)
        print(f"⚠️ No existing retrain model found for seed {seed}. Training from scratch...")
        if retrain_folder:
            print(f"   Searched in: {retrain_folder}")
        print("=" * 60)
        
        retain_dataset = retain_train_dl.dataset
        if isinstance(retain_dataset, torch.utils.data.Subset):
            base_dataset = retain_dataset.dataset
            indices = retain_dataset.indices
            shuffled_loader = DataLoader(
                Subset(base_dataset, indices),
                batch_size=retain_train_dl.batch_size,
                shuffle=True,
                num_workers=getattr(retain_train_dl, "num_workers", 4),
                pin_memory=True
            )
        else:
            shuffled_loader = DataLoader(
                retain_dataset,
                batch_size=retain_train_dl.batch_size,
                shuffle=True,
                num_workers=getattr(retain_train_dl, "num_workers", 4),
                pin_memory=True
            )

        batch_size = kwargs.get("batch_size", 256)
        warm_epochs = kwargs.get("warm", 1)
        lr = kwargs.get("lr", 0.1)

        # Get training schedule
        if model_name == "ViT":
            EPOCHS = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
            MILESTONES = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
        else:
            EPOCHS = getattr(conf, f"{dataset_name}_EPOCHS")
            MILESTONES = getattr(conf, f"{dataset_name}_MILESTONES")

        net.train()
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)

        iter_per_epoch = len(shuffled_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epochs)

        checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, "retrain", model_name, conf.TIME_NOW)
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, f"{model_name}-{dataset_name}-seed{seed}-ret{kwargs.get('ret_perc', 'NA')}-{{epoch}}-{{type}}.pth")

        best_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            if epoch > warm_epochs:
                train_scheduler.step(epoch)

            net.train()
            start = time.time()
            epoch_loss = 0.0
            correct_train = 0
            total_train_samples = 0

            for batch_index, (images, _, labels) in enumerate(shuffled_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size_actual = images.size(0)
                epoch_loss += loss.item() * batch_size_actual
                total_train_samples += batch_size_actual

                _, preds = outputs.max(1)
                correct_train += preds.eq(labels).sum().item()

                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * shuffled_loader.batch_size + batch_size_actual,
                    total_samples=len(shuffled_loader.dataset)
                ))

                if epoch <= warm_epochs:
                    warmup_scheduler.step()

            avg_train_loss = epoch_loss / total_train_samples
            train_acc = correct_train / total_train_samples
            finish = time.time()
            print("Epoch {} - Average Train Loss: {:.4f}, Train Accuracy: {:.4f}".format(epoch, avg_train_loss, train_acc))
            print("Epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))

            net.eval()
            test_loss = 0.0
            correct = 0.0
            with torch.no_grad():
                for images, _, labels in valid_dl:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    loss = loss_function(outputs, labels)
                    test_loss += loss.item()
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum()

            acc = correct.float() / len(valid_dl.dataset)
            print("Evaluating Network.....")
            print("Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
                epoch, test_loss / len(valid_dl.dataset), acc, time.time() - finish))

            if acc > best_acc:
                weights_path = checkpoint_path.format(epoch=epoch, type="best")
                print("Saving weights file to {}".format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc

    # Final metrics - FIXED: pass net instead of model
    return get_metric_scores(
        net,  # FIXED: was incorrectly passing `model` (baseline)
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
      
      retain_train_dl_original,
      retain_valid_dl_original,
      forget_train_dl_original,
      forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )



def random_modify_model(
    model,
    seed,
    dataset_name,
    model_name,
    noise_std=0.01,
):
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    save_dir = "C:/Users/aliev/Downloads/selective-synaptic-dampening-main/selective-synaptic-dampening-main/src/checkpoint/random_modify"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"random_modify_seed{seed}_{dataset_name}_{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Randomly modified model saved to: {save_path}")
    return model

def finetune(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
   retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
  
  dataset_name,
  model_name,
    device,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
          retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )


def teacher(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,

      retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
  dataset_name,
  model_name,
  
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(list(retain_train_dl.dataset), int(0.3 *len(retain_train_dl.dataset)))


    if model_name == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
          retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )

    
def amnesiac(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
      retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
    num_classes,
  dataset_name,
  model_name,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for x, _, clabel in forget_train_dl.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, _, rnd))

    for x, _, y in retain_train_dl.dataset:
        if kwargs.get("dataset_name", "Cifar10") == "MUCAC": 
            unlearning_trainset.append((x, _, torch.tensor(rnd)))
        else:
            unlearning_trainset.append((x, _, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
          retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )


def FisherForgetting(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
  
      retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
    num_classes,
  dataset_name,
  model_name,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
          retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )


def ssdtuning(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
  
      retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
  
  dataset_name,
  model_name,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd = ssd_.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = ssd.calc_importance(forget_train_dl)

    original_importances = ssd.calc_importance(full_train_dl)
    ssd.modify_weight(original_importances, sample_importances)
    
    return get_metric_scores(
        ssd.model,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
          retain_train_dl_original,
    retain_valid_dl_original,
    forget_train_dl_original,
    forget_valid_dl_original,
        device,
      dataset_name,
  model_name
    )