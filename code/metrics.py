"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
Code adapted from https://github.com/if-loops/selective-synaptic-dampening/tree/main/src
https://arxiv.org/abs/2308.07707

FIXED:
- Fixed class-biased truncation: now uses proper random sampling instead of [:min_size]
- Fixed hardcoded random_state=42: now derives seed from data hash for reproducibility with variation
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


def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


# ZRF/UnLearningScore https://arxiv.org/abs/2205.08096
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in forget_dl:
            x, y, cy = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)


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
    """
    Randomly subsample a tensor to target_size using the given seed.
    FIXED: Replaces simple truncation [:min_size] which caused class bias.
    """
    if len(tensor) <= target_size:
        return tensor
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(tensor))[:target_size]
    indices = np.sort(indices)  # Keep order for reproducibility
    return tensor[indices]


def _get_data_seed(tensor1, tensor2):
    """
    Derive a seed from the data tensors for reproducible but varying random states.
    FIXED: Replaces hardcoded random_state=42.
    """
    # Use hash of tensor shapes and first few values for seed
    seed_val = len(tensor1) * 1000 + len(tensor2)
    if len(tensor1) > 0:
        seed_val += int(tensor1[0].sum().item() * 1000) % 10000
    return seed_val % (2**31)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    min_size = min(len(retain_prob), len(test_prob))
    
    # FIXED: Use random sampling instead of simple truncation
    seed = _get_data_seed(retain_prob, test_prob)
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
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
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
    
    # FIXED: Use random sampling instead of simple truncation
    seed = _get_data_seed(prob_set1, prob_set2)
    prob_set1 = _random_subsample(prob_set1, min_size, seed)
    prob_set2 = _random_subsample(prob_set2, min_size, seed + 1)

    print(f"Set1 Distribution: {len(prob_set1)} samples")
    print(f"Set2 Distribution: {len(prob_set2)} samples")

    X = torch.cat([entropy(prob_set1), entropy(prob_set2)]).cpu().numpy().reshape(-1, 1)
    Y = np.concatenate([np.ones(len(prob_set1)), np.zeros(len(prob_set2))])

    return X, Y


def get_membership_attack_prob_our(loaderSet1, loaderSet2, model, seed=None):
    """
    FIXED: Added seed parameter for varying random_state instead of hardcoded 42.
    If seed is None, derives it from the data.
    """
    X, Y = get_membership_attack_data_our(loaderSet1, loaderSet2, model)

    # FIXED: Use varying random_state instead of hardcoded 42
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
    """
    Computes saliency maps for both sets and prepares data for MIA.
    """
    saliency_set1 = collect_saliency_maps(loaderSet1, model)
    saliency_set2 = collect_saliency_maps(loaderSet2, model)

    min_size = min(len(saliency_set1), len(saliency_set2))
    
    # FIXED: Use random sampling instead of simple truncation
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
    Y = np.concatenate([np.ones(min_size), np.zeros(min_size)])  # 1 for Set1, 0 for Set2

    return X, Y


def evaluate_mia_xgboost(loaderSet1, loaderSet2, model, seed=None):
    """
    Trains an XGBoost classifier to differentiate between saliency maps.
    FIXED: Added seed parameter for varying random_state.
    """
    X, Y = get_mia_data_saliency(loaderSet1, loaderSet2, model)

    # FIXED: Use varying random_state
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
