
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
import numpy as np
import os
from sklearn.metrics import classification_report
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from shadow_attack.shadow import *


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)



def calculate_accuracy(model, data_loader, device='cuda'):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct / total*100



def train_model_with_loader(model, train_loader, training_epochs=100, lr=0.01, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    for epochy in range(training_epochs):
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model




# Function to train a model
def train_model_with_raw_tensors(model, train_data, train_labels, epochs=100, lr=0.01,bs=128*2, device='cuda'):
    dataset= torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in train_loader:
            img, label = batch
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
    return model





def get_confidences_shadow_model(model, imgs, labels, device, batch_size=128):
    data_loader = DataLoader(CustomDataset(imgs, labels), batch_size=batch_size, shuffle=False)
    model.eval()
    feats = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits = model(images)          # <- logits, shape [B, num_classes]
            feats.append(logits.detach().cpu())
    return torch.cat(feats)





from types import MethodType

def ensure_feature_method(model: nn.Module) -> nn.Module:
    """
    Guarantee model.feature(x) exists. By default it returns forward logits.
    """
    if not hasattr(model, "feature"):
        def _feature(self, x):
            return self.forward(x)  # logits
        model.feature = MethodType(_feature, model)
    return model


def shadow_zone(shadow_imgs, shadow_labs, num_shadow_models=2, epochs=100, lr=0.01, device='cuda'):
    in_features, out_features = [], []

    for _ in range(num_shadow_models):
        indices = torch.randperm(shadow_imgs.size(0))
        half = len(indices) // 2
        shadow_imgs1, shadow_labs1 = shadow_imgs[indices[:half]], shadow_labs[indices[:half]]
        shadow_imgs2, shadow_labs2 = shadow_imgs[indices[half:]], shadow_labs[indices[half:]]

        # model_in
        model_in = ShadowModel().to(device)
        model_in = ensure_feature_method(model_in)     # <-- add
        model_in = train_model_with_raw_tensors(model_in, shadow_imgs1, shadow_labs1, epochs, lr, device=device)
        in_features.append(get_confidences_shadow_model(model_in, shadow_imgs1, shadow_labs1, device))
        out_features.append(get_confidences_shadow_model(model_in, shadow_imgs2, shadow_labs2, device))

        # model_out
        model_out = ShadowModel().to(device)
        model_out = ensure_feature_method(model_out)   # <-- add
        model_out = train_model_with_raw_tensors(model_out, shadow_imgs2, shadow_labs2, epochs, lr, device=device)
        in_features.append(get_confidences_shadow_model(model_out, shadow_imgs1, shadow_labs1, device))
        out_features.append(get_confidences_shadow_model(model_out, shadow_imgs2, shadow_labs2, device))

    in_features = torch.cat(in_features)
    out_features = torch.cat(out_features)
    return in_features, out_features



# Train the attack model
def attack_zone(target_data,
                target_model,
                in_features,
                out_features,
                num_epochs: int,
                lr: float,
                attack_hidden_size: int,
                device: str = 'cuda') -> float:
    """
    Train the attack classifier on shadow model logits and test it
    on the target sample.

    Parameters
    ----------
    target_data : torch.Tensor
        Single sample (C,H,W) from the measurement set.
    target_model : nn.Module
        The trained target model.
    in_features : torch.Tensor
        Shadow-model logits for member samples. Shape [N_in, num_classes].
    out_features : torch.Tensor
        Shadow-model logits for non-member samples. Shape [N_out, num_classes].
    num_epochs : int
        Attack classifier training epochs.
    lr : float
        Learning rate for the attack classifier.
    attack_hidden_size : int
        Size of the hidden layer in the attack MLP.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    float
        Membership probability for the target sample.
    """

    # ---------- Prepare dataset ----------
    X = torch.cat([in_features, out_features])        # shape [N, num_classes]
    y = torch.tensor(
        [1] * len(in_features) + [0] * len(out_features),
        dtype=torch.float32
    ).unsqueeze(1)

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # ---------- Target feature (logits) ----------
    target_model.eval()
    with torch.no_grad():
        target_feature = target_model(target_data.unsqueeze(0).to(device))
        # target_feature shape: [1, num_classes]
    D = target_feature.shape[-1]

    # ---------- Attack classifier ----------
    classifier = nn.Sequential(
        nn.Linear(D, attack_hidden_size),
        nn.ReLU(),
        nn.Linear(attack_hidden_size, 1),
        nn.Sigmoid()
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # ---------- Train ----------
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for feats, lbls in train_loader:
            feats = feats.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            preds = classifier(feats)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"[Attack] Epoch {epoch+1}/{num_epochs} "
                  f"Loss: {total_loss/len(train_loader):.4f}")

    # ---------- Score ----------
    classifier.eval()
    with torch.no_grad():
        score = classifier(target_feature).item()

    return score







def membership_inference(target_model, target_data, shadow_images, shadow_labels, num_shadow_models=2, shadow_epochs=100, shadow_lr=0.01,  attack_epochs=100, attack_lr=1e-3, attack_hidden_size=100, device='cuda'):
    in_features, out_features = shadow_zone(shadow_images, shadow_labels, num_shadow_models, shadow_epochs, shadow_lr, device)
    score = attack_zone(target_data, target_model, in_features, out_features, attack_epochs, attack_lr, attack_hidden_size, device)
    return score



def run_over_MIA(target_model, target_data_col, shadow_images, shadow_labels, num_shadow_models=2, shadow_epochs=100, shadow_lr=0.01,  attack_epochs=100, attack_lr=1e-3, attack_hidden_size=100, device='cuda'):
    result_col=[]
    for i in range(len(target_data_col)):
        target_data=target_data_col[i]
        result=membership_inference(target_model, target_data, shadow_images, shadow_labels, num_shadow_models, shadow_epochs, shadow_lr,  attack_epochs, attack_lr, attack_hidden_size, device)
        result_col.append(result)
    return np.array(result_col)
