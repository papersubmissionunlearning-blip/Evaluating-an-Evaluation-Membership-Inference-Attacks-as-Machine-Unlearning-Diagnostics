"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

FIXED:
- LiRA label inversion: train samples now correctly labeled as 1 (members), test as 0 (non-members)
- perc parameter: default changed from 0.0 to 0.2 to ensure shadow models have training data
- Consistent index file naming with title suffix
- Random sampling instead of truncation for balanced sets
"""
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from quantile.model import *
from quantile.quantile_attack import *


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, _, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def _random_subsample(tensor, target_size, seed):
    """Randomly subsample a tensor to target_size using the given seed."""
    if len(tensor) <= target_size:
        return tensor
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(tensor))[:target_size]
    indices = np.sort(indices)
    return tensor[indices]


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    min_size = min(len(retain_prob), len(test_prob))
    
    # FIXED: Use random sampling instead of simple truncation
    seed = len(retain_prob) * 1000 + len(test_prob)
    retain_prob = _random_subsample(retain_prob, min_size, seed)
    test_prob = _random_subsample(test_prob, min_size, seed + 1)
    
    print(f"retain_prob Distribution: {len(retain_prob)} samples")
    print(f"test_prob Distribution: {len(test_prob)} samples")
    print(f"forget_prob Distribution: {len(forget_prob)} samples")
    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


def get_membership_attack_data_our(loaderSet1, loaderSet2, model):
    prob_set1 = collect_prob(loaderSet1, model)
    prob_set2 = collect_prob(loaderSet2, model)

    min_size = min(len(prob_set1), len(prob_set2))
    
    # FIXED: Use random sampling
    seed = len(prob_set1) * 1000 + len(prob_set2)
    prob_set1 = _random_subsample(prob_set1, min_size, seed)
    prob_set2 = _random_subsample(prob_set2, min_size, seed + 1)

    print(f"Set1 Distribution: {len(prob_set1)} samples")
    print(f"Set2 Distribution: {len(prob_set2)} samples")

    X = torch.cat([entropy(prob_set1), entropy(prob_set2)]).cpu().numpy().reshape(-1, 1)
    Y = np.concatenate([np.ones(len(prob_set1)), np.zeros(len(prob_set2))])

    return X, Y


def get_membership_attack_prob_our(loaderSet1, loaderSet2, model, seed=None):
    """FIXED: Added seed parameter for varying random_state."""
    X, Y = get_membership_attack_data_our(loaderSet1, loaderSet2, model)

    if seed is None:
        seed = int(np.sum(X[:10]) * 1000) % (2**31) if len(X) > 0 else 42
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed, stratify=Y
    )

    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    return accuracy 


def get_mia_quantile(
    loader1,
    loader2,
    model,
    device,
    title,
    perc=0.2,  # FIXED: Changed from 0.0 to 0.2
    perc_test=0.20,
    measurement_number=10,
    auxillary_bs=32,
    n_quantile=100,
    low_quantile=0.01,
    high_quantile=0.99,
    use_logscale=False,
    use_gaussian=False,
    batch_size=32,
    quantile_model_epochs=30,
    quantile_model_lr=1e-3,
    alpha=0.05,
):
    """Run quantile-based MIA and return only AUC (float)."""

    # ---- Collect tensors from loaders ----
    train_images, train_labels = [], []
    for x, _, y in loader1:
        train_images.append(x)
        train_labels.append(y)
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images, test_labels = [], []
    for x, _, y in loader2:
        test_images.append(x)
        test_labels.append(y)
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)

    # ---- Create deterministic index files per run (FIXED: consistent naming) ----
    idx_path_train = Path(f"original_indices_{title}")
    if not idx_path_train.exists():
        original_indices = torch.randperm(len(train_images))
        torch.save(original_indices, idx_path_train.as_posix())
    else:
        original_indices = torch.load(idx_path_train.as_posix())

    idx_path_test = Path(f"original_indices_test_{title}")
    if not idx_path_test.exists():
        original_indices_test = torch.randperm(len(test_images))
        torch.save(original_indices_test, idx_path_test.as_posix())
    else:
        original_indices_test = torch.load(idx_path_test.as_posix())

    # ---- Split attacker/measurement samples ----
    num_samples_train = int(perc * len(train_images))
    num_samples_test = int(perc_test * len(test_images))
    
    print(f"Quantile MIA - Using {num_samples_train} train samples, {num_samples_test} test samples")

    indices_train = original_indices[:num_samples_train]
    anti_indices_train = original_indices[num_samples_train:num_samples_train + measurement_number]

    indices_test = original_indices_test[:num_samples_test]
    anti_indices_test = original_indices_test[num_samples_test:num_samples_test + measurement_number]

    attacker_train_images = train_images[indices_train]
    attacker_train_labels = train_labels[indices_train]
    measurement_train_images = train_images[anti_indices_train]
    measurement_train_labels = train_labels[anti_indices_train]

    attacker_test_images = test_images[indices_test]
    attacker_test_labels = test_labels[indices_test]
    measurement_test_images = test_images[anti_indices_test]
    measurement_test_labels = test_labels[anti_indices_test]

    # ---- Merge and prepare loaders ----
    shadow_images = torch.cat([attacker_train_images, attacker_test_images])
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels])
    measurement_images = torch.cat([measurement_train_images, measurement_test_images])
    measurement_labels = torch.cat([measurement_train_labels, measurement_test_labels])
    
    # FIXED: Correct labels - train samples are members (1), test samples are non-members (0)
    measurement_ref = np.array([1]*len(measurement_train_images) + [0]*len(measurement_test_images))

    aux_dataset = TensorDataset(shadow_images, shadow_labels)
    aux_loader = DataLoader(aux_dataset, batch_size=auxillary_bs, shuffle=True)

    # ---- Run quantile attack ----
    model = model.to('cpu')
    scores = mia_attack(
        model,
        measurement_images,
        measurement_labels,
        aux_loader,
        n_quantile=n_quantile,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        use_logscale=use_logscale,
        use_gaussian=use_gaussian,
        batch_size=batch_size,
        num_epochs=quantile_model_epochs,
        learning_rate=quantile_model_lr,
        alpha=alpha,
        device=device
    )

    # ---- Compute AUC ----
    fpr, tpr, _ = roc_curve(measurement_ref, scores)
    auc_value = float(auc(fpr, tpr))
    return auc_value

  
def get_mia_lira(
    loader1,
    loader2,
    model,
    device,
    title,
    # hyperparameters with defaults
    input_shape=(3, 32, 32),
    channel=3,
    num_classes=10,
    hidden_size=512,
    output_size=10,
    epochs=50,
    lr=1e-3,
    perc=0.2,  # FIXED: Changed from 0.0 to 0.2
    perc_test=0.20,
    measurement_number=10,
    num_shadow_models=10,
    lr_shadow_model=1e-3,
    epochs_shadow_model=20,
):
    # Clean up old index files for this title
    for name in (f"original_indices_{title}", f"original_indices_test_{title}"):
        Path(name).unlink(missing_ok=True)  
        
    # Treat loader1 as train_loader and loader2 as test_loader
    train_loader = loader1
    test_loader = loader2
    target_model = model

    # determine number of samples known to attacker
    num_samples_train = int(perc * len(train_loader.dataset))
    num_samples_test = int(perc_test * len(test_loader.dataset))
    print("----------------------------------")
    print("Attackers knowledge:")
    print(f"Training Dataset Info: {num_samples_train}/{len(train_loader.dataset)} = {num_samples_train/len(train_loader.dataset)*100:.2f}%")
    print(f"Testing Dataset Info: {num_samples_test}/{len(test_loader.dataset)} = {num_samples_test/len(test_loader.dataset)*100:.2f}%")
    print("----------------------------------")

    # load all train images/labels into tensors
    train_images = []
    train_labels = []
    for images, _, labels in train_loader:
        train_images.append(images)
        train_labels.append(labels)
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    # FIXED: Consistent index file naming with title
    idx_path = f'original_indices_{title}'
    if not os.path.exists(idx_path):
        original_indices = torch.randperm(len(train_images))
        torch.save(original_indices, idx_path)
    else:
        original_indices = torch.load(idx_path)

    indices = original_indices[:num_samples_train]
    anti_indices = original_indices[num_samples_train:num_samples_train + measurement_number]

    attacker_train_images = train_images[indices]
    attacker_train_labels = train_labels[indices]
    measurement_train_images = train_images[anti_indices]
    measurement_train_labels = train_labels[anti_indices]

    # load all test images/labels into tensors
    test_images = []
    test_labels = []
    for images, _, labels in test_loader:
        test_images.append(images)
        test_labels.append(labels)
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)

    # FIXED: Consistent index file naming with title
    idx_path_test = f'original_indices_test_{title}'
    if not os.path.exists(idx_path_test):
        original_indices_test = torch.randperm(len(test_images))
        torch.save(original_indices_test, idx_path_test)
    else:
        original_indices_test = torch.load(idx_path_test)

    indices = original_indices_test[:num_samples_test]
    anti_indices = original_indices_test[num_samples_test:num_samples_test + measurement_number]

    attacker_test_images = test_images[indices]
    attacker_test_labels = test_labels[indices]
    measurement_test_images = test_images[anti_indices]
    measurement_test_labels = test_labels[anti_indices]

    # shadow set is concatenation of attacker's known train and test subsets
    shadow_images = torch.cat([attacker_train_images, attacker_test_images])
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels])

    # measurement set is the held-out examples (some from train, some from test)
    measurement_images = torch.cat([measurement_train_images, measurement_test_images])
    measurement_labels = torch.cat([measurement_train_labels, measurement_test_labels])
    
    # FIXED: Correct labels - train samples are members (1), test samples are non-members (0)
    # Original code had this inverted: [0]*train + [1]*test which is WRONG
    measurement_ref = np.array([1] * len(measurement_train_images) + [0] * len(measurement_test_images))

    print(f"Measurement Sample Size: {len(measurement_images)}")
    print(f"Members (train): {len(measurement_train_images)}, Non-members (test): {len(measurement_test_images)}")

    # -------- LIRA -----------------------------------------------------------------
    # shadow zone: estimate loss distributions (uses your helper)
    in_mean, in_std, out_mean, out_std = estimate_loss_distributions(
        measurement_images,
        measurement_labels,
        shadow_images,
        shadow_labels,
        num_shadow_models=num_shadow_models,
        epochs=epochs_shadow_model,
        lr=lr_shadow_model
    )

    # attack zone: compute MIA scores for measurement samples (uses your helper)
    scores = run_over_MIA(
        target_model,
        measurement_images,
        measurement_labels,
        in_mean,
        in_std,
        out_mean,
        out_std
    )

    # compute ROC and AUC
    fpr, tpr, _thresholds = roc_curve(measurement_ref, scores)
    auc_value = auc(fpr, tpr)
    print(f"LiRA AUC: {auc_value}")
    return float(auc_value)

  
def get_mia_shadow(
    loader1,                    # DataLoader (e.g., training set)
    loader2,                    # DataLoader (e.g., testing set)
    target_model,               # The model to attack
    device,                     # torch.device("cuda" or "cpu")
    title="shadow",             # FIXED: Added title parameter
    perc=0.2,                   # FIXED: Changed from 0.0 to 0.2
    perc_test=0.2,
    measurement_number=20,
    num_shadow_models=5,
    shadow_epochs=30,
    shadow_lr=1e-3,
    attack_epochs=50,
    attack_lr=1e-3,
    attack_hidden_size=128,
    verbose=True
):
    # Clean up old index files
    for name in (f"original_indices_{title}", f"original_indices_test_{title}"):
        Path(name).unlink(missing_ok=True)  
        
    # ------------------- Prepare attacker knowledge -------------------
    num_samples_train = int(perc * len(loader1.dataset))
    num_samples_test = int(perc_test * len(loader2.dataset))

    if verbose:
        print("----------------------------------")
        print("Attackers knowledge:")
        print(f"Training Dataset Info: {num_samples_train}/{len(loader1.dataset)} = "
              f"{(num_samples_train / len(loader1.dataset) * 100 if len(loader1.dataset) else 0):.2f}%")
        print(f"Testing Dataset Info: {num_samples_test}/{len(loader2.dataset)} = "
              f"{(num_samples_test / len(loader2.dataset) * 100 if len(loader2.dataset) else 0):.2f}%")
        print("----------------------------------")

    # ------------------- Gather data tensors -------------------
    def gather_images_labels(loader):
        images, labels = [], []
        for img, _, lbl in loader:
            images.append(img)
            labels.append(lbl)
        return torch.cat(images), torch.cat(labels)

    train_images, train_labels = gather_images_labels(loader1)
    test_images, test_labels = gather_images_labels(loader2)

    # ------------------- Fix indices for reproducibility (FIXED: consistent naming) -------------------
    def get_indices(filename, total_len):
        if not os.path.exists(filename):
            idx = torch.randperm(total_len)
            torch.save(idx, filename)
        else:
            idx = torch.load(filename)
        return idx

    train_idx = get_indices(f"original_indices_{title}", len(train_images))
    test_idx = get_indices(f"original_indices_test_{title}", len(test_images))

    attacker_train_images = train_images[train_idx[:num_samples_train]]
    attacker_train_labels = train_labels[train_idx[:num_samples_train]]
    measurement_train_images = train_images[train_idx[num_samples_train:num_samples_train + measurement_number]]
    measurement_train_labels = train_labels[train_idx[num_samples_train:num_samples_train + measurement_number]]

    attacker_test_images = test_images[test_idx[:num_samples_test]]
    attacker_test_labels = test_labels[test_idx[:num_samples_test]]
    measurement_test_images = test_images[test_idx[num_samples_test:num_samples_test + measurement_number]]
    measurement_test_labels = test_labels[test_idx[num_samples_test:num_samples_test + measurement_number]]

    # ------------------- Combine shadow and measurement sets -------------------
    shadow_images = torch.cat([attacker_train_images, attacker_test_images])
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels])
    measurement_images = torch.cat([measurement_train_images, measurement_test_images])
    
    # FIXED: Correct labels - train samples are members (1), test samples are non-members (0)
    measurement_ref = np.array([1] * len(measurement_train_images) + [0] * len(measurement_test_images))

    if verbose:
        print(f"Measurement Sample Size: {len(measurement_images)}")
        print(f"Members (train): {len(measurement_train_images)}, Non-members (test): {len(measurement_test_images)}")

    # ------------------- Run Membership Inference Attack -------------------
    scores = run_over_MIA(
        target_model,
        measurement_images,
        shadow_images,
        shadow_labels,
        num_shadow_models,
        shadow_epochs,
        shadow_lr,
        attack_epochs,
        attack_lr,
        attack_hidden_size,
        device
    )

    # ------------------- Compute AUC -------------------
    fpr, tpr, _ = roc_curve(measurement_ref, scores)
    roc_auc = auc(fpr, tpr)
    
    if verbose:
        print(f"Shadow MIA AUC: {roc_auc}")

    return roc_auc


#--------SALIENCY SETUP START --------------------

import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms


def collect_saliency_maps(data_loader, model):
    data_loader = DataLoader(data_loader.dataset, batch_size=1, shuffle=False)

    model.eval()
    saliency_maps = []
    
    for batch in data_loader:
        batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
        data, _, target = batch 

        data.requires_grad_()  
        model.zero_grad()

        outputs = model(data)

        scores, _ = torch.max(outputs, dim=1)
        scores = scores.sum()  
        scores.backward()  

        saliency = data.grad.abs().mean(dim=1)  
        saliency_maps.append(saliency.cpu().numpy())

    return np.concatenate(saliency_maps, axis=0)


def get_mia_data_saliency(loaderSet1, loaderSet2, model):
    """Computes saliency maps for both sets and prepares data for MIA."""
    saliency_set1 = collect_saliency_maps(loaderSet1, model)
    saliency_set2 = collect_saliency_maps(loaderSet2, model)

    min_size = min(len(saliency_set1), len(saliency_set2))
    
    # FIXED: Use random sampling
    seed = len(saliency_set1) * 1000 + len(saliency_set2)
    rng = np.random.RandomState(seed % (2**31))
    
    if len(saliency_set1) > min_size:
        idx1 = np.sort(rng.permutation(len(saliency_set1))[:min_size])
        saliency_set1 = saliency_set1[idx1]
    if len(saliency_set2) > min_size:
        idx2 = np.sort(rng.permutation(len(saliency_set2))[:min_size])
        saliency_set2 = saliency_set2[idx2]

    print(f"Set1 Distribution: {len(saliency_set1)} samples")
    print(f"Set2 Distribution: {len(saliency_set2)} samples")

    X = np.vstack([saliency_set1.reshape(min_size, -1), saliency_set2.reshape(min_size, -1)])
    Y = np.concatenate([np.ones(min_size), np.zeros(min_size)])

    return X, Y


def evaluate_mia_xgboost(loaderSet1, loaderSet2, model, seed=None):
    """Trains an XGBoost classifier to differentiate between saliency maps."""
    X, Y = get_mia_data_saliency(loaderSet1, loaderSet2, model)

    if seed is None:
        seed = int(np.sum(X[:10, :10]) * 100) % (2**31) if len(X) > 0 else 42
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed, stratify=Y
    )

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed
    )

    xgb_clf.fit(X_train, Y_train)

    Y_pred = xgb_clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(f"MIA Attack Accuracy with XGBoost: {accuracy:.4f}")
    return accuracy

#--------SALIENCY SETUP END --------------------


@torch.no_grad()
def actv_dist(model1, model2, dataloader, device="cuda"):
    sftmx = nn.Softmax(dim=1)
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(
            torch.sum(
                torch.square(
                    F.softmax(model1_out, dim=1) - F.softmax(model2_out, dim=1)
                ),
                axis=1,
            )
        )
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis=0)
    return distances.mean()
