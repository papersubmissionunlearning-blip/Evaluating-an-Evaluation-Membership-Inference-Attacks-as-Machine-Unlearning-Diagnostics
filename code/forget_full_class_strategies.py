"""
FIXED: retrain() now loads pre-existing retrain models from disk if available
"""
import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
import itertools

from tqdm import tqdm
from sklearn import linear_model, model_selection
from collections import OrderedDict


from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob, get_membership_attack_prob_our, evaluate_mia_xgboost
from utils import *
import ssd as ssd
import conf
import timeit
from retrain_utils import find_best_retrain_model

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


from collections import Counter

def get_size_dl(name, dl):
    print(name, str(len(dl.dataset)))

# def get_classwise_ds(ds, num_classes):
#     classwise_ds = {i: [] for i in range(num_classes)}
#     for img, label, clabel in ds:
#         classwise_ds[clabel].append((img, label, clabel))
#     return classwise_ds


# def build_retain_forget_sets(
#     classwise_train,
#     classwise_test,
#     num_classes,
#     forget_class,
#     percent_to_transfer: int = 0,  # e.g., 0, 25, 100
# ):
#     """
#     Build retain/forget splits. All non-forget classes go to RETAIN.
#     For the 'forget_class', transfer the FIRST k% (sequentially) of its samples to RETAIN,
#     and keep the remaining (100 - k)% in FORGET.

#     Examples:
#       - percent_to_transfer=0   -> 0% moved; 100% stay in FORGET.
#       - percent_to_transfer=25  -> first 25% moved to RETAIN; 75% stay in FORGET.
#       - percent_to_transfer=100 -> all moved to RETAIN; FORGET gets 0.
#     """
#     # clamp to [0, 100]
#     p = max(0, min(int(percent_to_transfer), 100))

#     def split_sequential(samples):
#         n = len(samples)
#         k = int(round((p / 100.0) * n))
#         k = max(0, min(k, n))
#         to_transfer = samples[:k]   # FIRST k% go to RETAIN
#         to_forget   = samples[k:]   # remaining stay in FORGET
#         return to_transfer, to_forget

#     # VALID
#     forget_valid, retain_valid = [], []
#     for cls in range(num_classes):
#         samples = classwise_test[cls]
#         if cls == forget_class:
#             to_transfer, to_forget = split_sequential(samples)
#             retain_valid.extend(to_transfer)
#             forget_valid.extend(to_forget)
#         else:
#             retain_valid.extend(samples)

#     # TRAIN
#     forget_train, retain_train = [], []
#     for cls in range(num_classes):
#         samples = classwise_train[cls]
#         if cls == forget_class:
#             to_transfer, to_forget = split_sequential(samples)
#             retain_train.extend(to_transfer)
#             forget_train.extend(to_forget)
#         else:
#             retain_train.extend(samples)

#     # ---- class distributions print ----
#     def count_classes(data):
#         cnt = Counter()
#         for _, _, clabel in data:
#             cnt[clabel] += 1
#         return [cnt.get(c, 0) for c in range(num_classes)]

#     print("\nClass distributions (counts per class):")
#     print("Retain TRAIN :", count_classes(retain_train))
#     print("Retain VALID :", count_classes(retain_valid))
#     print("Forget TRAIN :", count_classes(forget_train))
#     print("Forget VALID :", count_classes(forget_valid))

#     return (retain_train, retain_valid, forget_train, forget_valid)

# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    """
    Group a dataset into a dict {class_id: [samples…]} and
    preserve the original dataset index for each sample.

    Returns:
        dict[int, list[tuple]] where each element is
        (idx, img, label, clabel)
        so downstream code can extract idx directly.
    """
    classwise_ds = {i: [] for i in range(num_classes)}
    for idx, (img, label, clabel) in enumerate(ds):
        # keep idx as the first element
        classwise_ds[clabel].append((idx, img, label, clabel))
    return classwise_ds


from collections import Counter



def build_retain_forget_sets(
    classwise_train,
    classwise_test,
    num_classes,
    forget_class,
    percent_to_transfer: int = 0,  # 0..100, sequential (first k%)
):
    """
    Returns:
        (retain_train_idx, retain_valid_idx,
         forget_train_idx, forget_valid_idx)

    Also prints class distributions based on clabels.
    Assumes every sample is (idx, img, label, clabel)
    or similar, with idx first and clabel last.
    """
    p = max(0, min(int(percent_to_transfer), 100))

    def extract_idx(sample):
        if isinstance(sample, (tuple, list)) and isinstance(sample[0], int):
            return sample[0]
        raise ValueError("Each sample must begin with its dataset index.")

    def split_indices(samples):
        n = len(samples)
        k = int(round((p / 100.0) * n))
        all_idx = [extract_idx(s) for s in samples]
        return all_idx[:k], all_idx[k:]

    retain_valid_idx, forget_valid_idx = [], []
    retain_train_idx, forget_train_idx = [], []

    retain_valid_samples, forget_valid_samples = [], []
    retain_train_samples, forget_train_samples = [], []

    # VALID
    for cls in range(num_classes):
        s = classwise_test[cls]
        if cls == forget_class:
            to_ret, to_fgt = split_indices(s)
            retain_valid_idx += to_ret
            forget_valid_idx += to_fgt
            retain_valid_samples += s[:len(to_ret)]
            forget_valid_samples += s[len(to_ret):]
        else:
            retain_valid_idx += [extract_idx(x) for x in s]
            retain_valid_samples += s

    # TRAIN
    for cls in range(num_classes):
        s = classwise_train[cls]
        if cls == forget_class:
            to_ret, to_fgt = split_indices(s)
            retain_train_idx += to_ret
            forget_train_idx += to_fgt
            retain_train_samples += s[:len(to_ret)]
            forget_train_samples += s[len(to_ret):]
        else:
            retain_train_idx += [extract_idx(x) for x in s]
            retain_train_samples += s

    # ---- print stats ----
    print(f"\nSplit (percent_to_transfer={p}%):")
    print(f"Retain TRAIN: {len(retain_train_idx)} | Forget TRAIN: {len(forget_train_idx)}")
    print(f"Retain VALID: {len(retain_valid_idx)} | Forget VALID: {len(forget_valid_idx)}")

    def count_classes(data):
        cnt = Counter()
        for sample in data:
            clabel = sample[-1]            # last element is clabel
            cnt[clabel] += 1
        return [cnt.get(c, 0) for c in range(num_classes)]

    print("\nClass distributions (counts per class):")
    print("Retain TRAIN :", count_classes(retain_train_samples))
    print("Retain VALID :", count_classes(retain_valid_samples))
    print("Forget TRAIN :", count_classes(forget_train_samples))
    print("Forget VALID :", count_classes(forget_valid_samples))

    return (
        retain_train_idx,
        retain_valid_idx,
        forget_train_idx,
        forget_valid_idx,
    )
  
# def build_retain_forget_sets(
#     classwise_train, classwise_test, num_classes, forget_class
# ):
#     # Getting the forget and retain validation data
#     forget_valid = []
#     for cls in range(num_classes):
#         if cls == forget_class:
#             for img, label, clabel in classwise_test[cls]:
#                 forget_valid.append((img, label, clabel))

#     retain_valid = []
#     for cls in range(num_classes):
#         if cls != forget_class:
#             for img, label, clabel in classwise_test[cls]:
#                 retain_valid.append((img, label, clabel))

#     forget_train = []
#     for cls in range(num_classes):
#         if cls == forget_class:
#             for img, label, clabel in classwise_train[cls]:
#                 forget_train.append((img, label, clabel))

#     retain_train = []
#     for cls in range(num_classes):
#         if cls != forget_class:
#             for img, label, clabel in classwise_train[cls]:
#                 retain_train.append((img, label, clabel))

#     # ---- NEW: print class distributions ----
#     def count_classes(data):
#         cnt = Counter()
#         for _, _, clabel in data:
#             cnt[clabel] += 1
#         return [cnt.get(c, 0) for c in range(num_classes)]

#     print("\nClass distributions")
#     print("Retain TRAIN :", count_classes(retain_train))
#     print("Retain VALID :", count_classes(retain_valid))
#     print("Forget TRAIN :", count_classes(forget_train))
#     print("Forget VALID :", count_classes(forget_valid))

#     return (retain_train, retain_valid, forget_train, forget_valid)



from collections import Counter





# Returns metrics
def get_metric_scores(
    model,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
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
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model) 
    
    mia_forget_retain = get_membership_attack_prob_our(retain_train_dl, forget_train_dl, model)
    mia_forget_test = get_membership_attack_prob_our(valid_dl, forget_train_dl, model)
    mia_retain_test = get_membership_attack_prob_our(retain_train_dl, valid_dl, model)
    mia_train_test = get_membership_attack_prob_our(valid_dl, train_dl, model)
    
    # mia_forget_retain = evaluate_mia_xgboost(retain_train_dl, forget_train_dl, model)
    # mia_forget_test = evaluate_mia_xgboost(valid_dl, forget_train_dl, model)
    # mia_retain_test = evaluate_mia_xgboost(retain_train_dl, valid_dl, model)
    # mia_train_test = evaluate_mia_xgboost(valid_dl, train_dl, model)
    
    # loss_acc_dict =  0
    # retain_acc_dict = 0
    # zrf = 0
    # d_f = 0
    # mia = 0
    

    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, mia_forget_retain, mia_forget_test, mia_retain_test, mia_train_test, d_f["Acc"]) 


# Does nothing; original model
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
        device,
    )



# Retrain the model on the retrain dataset only
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

        checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, "retrain_fullclass", model_name, conf.TIME_NOW)
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

    # Final metrics
    return get_metric_scores(
        net,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


# Finetune the model using the retain data for a set number of epochs
def finetune(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
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
        device,
    )

# Bad Teacher from https://github.com/vikram2000b/bad-teaching-unlearning
def teacher(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(list(retain_train_dl.dataset), int(0.3 *len(retain_train_dl.dataset)))


    if kwargs["model_name"] == "ViT":
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
        device,
    )


# Implementation from https://github.com/vikram2000b/bad-teaching-unlearning
def amnesiac(
    model,  seed,
    unlearning_teacher,
    train_dl,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
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
        device,
    )


# Extremely slow >>> Fisher https://github.com/AdityaGolatkar/SelectiveForgetting
def NTK(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    **kwargs,
):
    def delta_w_utils(model_init, dataloader, name="complete"):
        model_init.eval()
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False
        )
        G_list = []
        f0_minus_y = []
        for idx, batch in enumerate(
            tqdm(dataloader)
        ):  # (tqdm(dataloader,leave=False)):
            batch = [
                tensor.to(next(model_init.parameters()).device) for tensor in batch
            ]
            input, _, target = batch

            target = target.cpu().detach().numpy()
            output = model_init(input)
            G_sample = []
            for cls in range(num_classes):
                grads = torch.autograd.grad(
                    output[0, cls], model_init.parameters(), retain_graph=True
                )
                grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
                G_sample.append(grads)
                G_list.append(grads)
                p = (
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                    .transpose()
                )
                p[target] -= 1
                f0_y_update = deepcopy(p)
            f0_minus_y.append(f0_y_update)
        return np.stack(G_list).transpose(), np.vstack(f0_minus_y)

    #############################################################################################
    model_init = deepcopy(model)
    G_r, f0_minus_y_r = delta_w_utils(deepcopy(model), retain_train_dl, "complete")
    print("GOT GR")
    # np.save('NTK_data/G_r.npy',G_r)
    # np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
    # del G_r, f0_minus_y_r

    G_f, f0_minus_y_f = delta_w_utils(deepcopy(model), forget_train_dl, "retain")
    print("GOT GF")
    # np.save('NTK_data/G_f.npy',G_f)
    # np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
    # del G_f, f0_minus_y_f

    # G_r = np.load('NTK_data/G_r.npy')
    # G_f = np.load('NTK_data/G_f.npy')
    G = np.concatenate([G_r, G_f], axis=1)
    print("GOT G")
    # np.save('NTK_data/G.npy',G)
    # del G, G_f, G_r

    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    # f0_minus_y_f = np.load('NTK_data/f0_minus_y_f.npy')
    f0_minus_y = np.concatenate([f0_minus_y_r, f0_minus_y_f])

    # np.save('NTK_data/f0_minus_y.npy',f0_minus_y)
    # del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    weight_decay = 0.1

    # G = np.load('NTK_data/G.npy')
    theta = G.transpose().dot(G) + (
        len(retain_train_dl.dataset) + len(forget_train_dl.dataset)
    ) * weight_decay * np.eye(G.shape[1])
    # del G

    theta_inv = np.linalg.inv(theta)

    # np.save('NTK_data/theta.npy',theta)
    # del theta

    # G = np.load('NTK_data/G.npy')
    # f0_minus_y = np.load('NTK_data/f0_minus_y.npy')
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    # np.save('NTK_data/theta_inv.npy',theta_inv)
    # np.save('NTK_data/w_complete.npy',w_complete)
    # del G, f0_minus_y, theta_inv, w_complete

    # G_r = np.load('NTK_data/G_r.npy')
    num_to_retain = len(retain_train_dl.dataset)
    theta_r = G_r.transpose().dot(G_r) + num_to_retain * weight_decay * np.eye(
        G_r.shape[1]
    )
    # del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    # np.save('NTK_data/theta_r.npy',theta_r)
    # del theta_r

    # G_r = np.load('NTK_data/G_r.npy')
    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    # np.save('NTK_data/theta_r_inv.npy',theta_r_inv)
    # np.save('NTK_data/w_retain.npy',w_retain)
    # del G_r, f0_minus_y_r, theta_r_inv, w_retain

    def get_delta_w_dict(delta_w, model):
        # Give normalized delta_w
        delta_w_dict = OrderedDict()
        params_visited = 0
        for k, p in model.named_parameters():
            num_params = np.prod(list(p.shape))
            update_params = delta_w[params_visited : params_visited + num_params]
            delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
            params_visited += num_params
        return delta_w_dict

    #### Scrubbing Direction
    # w_complete = np.load('NTK_data/w_complete.npy')
    # w_retain = np.load('NTK_data/w_retain.npy')
    print("got prelims, calculating delta_w")
    delta_w = (w_retain - w_complete).squeeze()
    print("got delta_w")
    # delta_w_copy = deepcopy(delta_w)
    # delta_w_actual = vectorize_params(model0)-vectorize_params(model)

    # print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    # print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    # scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    # print('Actual Scale: {}'.format(scale_ratio))
    # log_dict['actual_scale_ratio']=scale_ratio
    def vectorize_params(model):
        param = []
        for p in model.parameters():
            param.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(param)

    m_pred_error = (
        vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()
    )
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(
        delta_w / np.linalg.norm(delta_w), m_pred_error / np.linalg.norm(m_pred_error)
    )
    print(f"Inner Product--: {inner}")

    if inner < 0:
        angle = np.arccos(inner) - np.pi / 2
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.sin(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner)
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.cos(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale = predicted_norm / np.linalg.norm(delta_w)
    predicted_scale
    print(f"Predicted Scale:  {predicted_scale}")
    # log_dict['predicted_scale_ratio']=predicted_scale

    # def NIP(v1,v2):
    #     nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    #     print(nip)
    #     return nip
    # nip=NIP(delta_w_actual,delta_w)
    # log_dict['nip']=nip
    scale = predicted_scale
    direction = get_delta_w_dict(delta_w, model)

    for k, p in model.named_parameters():
        p.data += (direction[k] * scale).to(device)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


# From https://github.com/AdityaGolatkar/SelectiveForgetting
def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
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
        if p.size(0) == num_classes:
            mu[forget_class] = 0
            var[forget_class] = 0.0001
        if p.size(0) == num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
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
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), num_classes
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_classes):
        if i != forget_class:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_class
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise,
        forget_class_label,
        retain_samples,
        batch_size=noise_batch_size,
        device=device,
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = fit_one_unlearning_cycle(
        1, model, heal_loader, retain_valid_dl, device=device, lr=0.0001
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


# Ours
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
        device,
    )
