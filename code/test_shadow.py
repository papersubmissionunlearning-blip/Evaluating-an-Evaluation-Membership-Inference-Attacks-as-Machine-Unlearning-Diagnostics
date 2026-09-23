
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc
import models
import datasets
import pandas as pd
# Local imports
from shadow_attack.shadow_attack import run_over_MIA, train_model_with_loader, calculate_accuracy
from collections import Counter

# -----------------------------------------------------------------------------------
# Hyperparameters and constants
input_shape = (3, 32, 32)
channel = 3
num_classes = 10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
perc = 0.0        # amount of actual training data available to the attacker
perc_test = 0.20  # amount of testing data available to the attacker (similar distribution to training data)
measurement_number = 20  # number of target samples to be measured from each training and non-training data
num_shadow_models = 5
lr_shadow_model = 1e-3
epochs_shadow_model = 30
lr_attack_model = 1e-3
epochs_attack_model = 50
attack_hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model / training configuration (no argparse)
net_name = "ResNet18"  # must match torchvision.models attribute name
classes = 10
batch_size = 256
# default weight path in current working directory
current_dir = os.getcwd()
weight_path = os.path.join(current_dir, "checkpoint", "ResNet18", "ResNet18-Cifar10-20-best.pth")
# -----------------------------------------------------------------------------------

def transfer_percent_from_forget_to_retain(trainset, forget_indices, retain_indices, percent_to_transfer, num_classes=10, dataset_name = "Cifar10"):
    """
    Moves the first X% (per class) of forget_indices into retain_indices.
    Returns updated forget_indices, retain_indices.
    """
    if dataset_name == 'MUCAC':
        targets = np.array(trainset.labels)
    else:
        targets = np.array(trainset.targets)
    updated_forget = []
    updated_retain = list(retain_indices)  # make a copy

    forget_by_class = {c: [] for c in range(num_classes)}
    for idx in forget_indices:
        label = targets[idx]
        forget_by_class[label].append(idx)

    for class_label in range(num_classes):
        class_forget = forget_by_class[class_label]
        class_forget = sorted(class_forget)  # ensure consistent order

        num_to_transfer = int(len(class_forget) * percent_to_transfer)
        to_transfer = class_forget[:num_to_transfer]
        to_keep = class_forget[num_to_transfer:]

        updated_retain.extend(to_transfer)
        updated_forget.extend(to_keep)

    # Print updated class distributions
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
  
def load_forget_retain_indices(trainset, dataset_name, seed, forget_per_class, csv_path="split_indices.csv", num_classes=10):
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
        print(f"Class {c}: {retain_dist[c]}")

    print(f"\nForget class distribution for seed {seed}:")
    if dataset_name == "Mnist":
       forget_dist = Counter([label.item() for label in forget_labels])
    else: 
      forget_dist = Counter(forget_labels)
    for c in range(num_classes): 
        print(f"Class {c}: {forget_dist[c]}")
        
    return forget_indices, retain_indices

input_shape = (3, 32, 32)
channel = 3
num_classes = 10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
perc = 0.0        # amount of actual training data available to the attacker
perc_test = 0.20  # amount of testing data available to the attacker (similar distribution to training data)
measurement_number = 20  # number of target samples to be measured from each training and non-training data
num_shadow_models = 5
lr_shadow_model = 1e-3
epochs_shadow_model = 30
lr_attack_model = 1e-3
epochs_attack_model = 50
attack_hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32
# Model / training configuration (no argparse)
net_name = "ResNet18"  # must match torchvision.models attribute name
classes = 10
batch_size = 256
# default weight path in current working directory
current_dir = os.getcwd()
weight_path = os.path.join(current_dir, "checkpoint", "ResNet18", "ResNet18-Cifar10-20-best.pth")

def main():
    # Ensure results folder
    os.makedirs('results', exist_ok=True)

    # ------------------- Dataset -------------------
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #    transforms.Lambda(lambda x: x.view(3, 32, 32))
    #])
    #train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------- Target model -------------------
    # Dynamically create the model and set number of classes
    root = "./data"
    trainset = getattr(datasets, "Cifar10")(
        root=root, download=True, train=True, unlearning=True, img_size=img_size
    )
    validset = getattr(datasets, "Cifar10")(
        root=root, download=True, train=False, unlearning=True, img_size=img_size
    )

    trainloader = DataLoader(trainset, num_workers=4, batch_size=256, shuffle=True)
    validloader = DataLoader(validset, num_workers=4, batch_size=256, shuffle=False)
    train_loader = trainloader
    test_loader = validloader
    
    forget_indices, retain_indices = load_forget_retain_indices(
        trainset=trainset,
        dataset_name='Cifar10',
        seed=1,
        forget_per_class=500, #change cifar 10 (500),   cifar 20 (50), MNIST (600), MUCAC (527)
        csv_path="splits/split_indices_Cifar10.csv",
        num_classes=10
    )
    
    forget_indices, retain_indices = transfer_percent_from_forget_to_retain(
       trainset=trainset,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        percent_to_transfer=(0/100),  # Move x% of forget set per class to retain set
        num_classes=10,
        dataset_name='Cifar10',
    )
    retainset = getattr(datasets, "Cifar10")(
        root=root,
        download=True,
        train=True,
        unlearning=False,
        img_size=img_size,
        indices=retain_indices
    )
    forgetset = getattr(datasets, "Cifar10")(
        root=root,
        download=True,
        train=True,
        unlearning=False,
        img_size=img_size,
        indices=forget_indices
    )

    
    retainset2 = getattr(datasets, "Cifar10")(
        root=root,
        download=True,
        train=True,
        unlearning=True,
        img_size=img_size,
        indices=retain_indices
    )
    forgetset2 = getattr(datasets, "Cifar10")(
        root=root,
        download=True,
        train=True,
        unlearning=True,
        img_size=img_size,
        indices=forget_indices
    )
    
    retain_train_dl = DataLoader(retainset, batch_size=256, shuffle=True)  
    forget_train_dl = DataLoader(forgetset, batch_size=256, shuffle=False)    

    forget_valid_dl = forget_train_dl
    retain_valid_dl = retain_train_dl
    
    target_model = getattr(models, net_name)(num_classes=classes).to(device)

    # Load or train
    if weight_path and os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        state = torch.load(weight_path, map_location=device)
        target_model.load_state_dict(state)
    else:
        print(f"No weights found at {weight_path}. Training {net_name} from scratch for {epochs} epochs.")
        # Create directory for checkpoint if provided
        if weight_path:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        train_model_with_loader(target_model, train_loader, epochs, lr, device)
        if weight_path:
            torch.save(target_model.state_dict(), weight_path)

    # ------------------- Evaluate target -------------------


    # ------------------- Prepare attacker knowledge -------------------
    num_samples_train = int(perc * len(train_loader.dataset))
    num_samples_test = int(perc_test * len(test_loader.dataset))
    print("----------------------------------")
    print("Attackers knowledge:")
    print(f"Training Dataset Info: {num_samples_train}/{len(train_loader.dataset)} = "
          f"{(num_samples_train / len(train_loader.dataset) * 100 if len(train_loader.dataset) else 0):.2f}%")
    print(f"Testing Dataset Info: {num_samples_test}/{len(test_loader.dataset)} = "
          f"{(num_samples_test / len(test_loader.dataset) * 100 if len(test_loader.dataset) else 0):.2f}%")
    print("----------------------------------")

    # Gather tensors for train set
    train_images, train_labels = [], []
    for images,  _, labels in train_loader:
        train_images.append(images)
        train_labels.append(labels)
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    # fixed random indices for reproducibility across runs
    if not os.path.exists('original_indices'):
        original_indices = torch.randperm(len(train_images))
        torch.save(original_indices, 'original_indices')
    else:
        original_indices = torch.load('original_indices')

    indices = original_indices[:num_samples_train]
    anti_indices = original_indices[num_samples_train:num_samples_train + measurement_number]
    attacker_train_images = train_images[indices]
    attacker_train_labels = train_labels[indices]
    measurement_train_images = train_images[anti_indices]
    measurement_train_labels = train_labels[anti_indices]

    # Gather tensors for test set
    test_images, test_labels = [], []
    for images,  _, labels in test_loader:
        test_images.append(images)
        test_labels.append(labels)
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)

    if not os.path.exists('original_indices_test'):
        original_indices_test = torch.randperm(len(test_images))
        torch.save(original_indices_test, 'original_indices_test')
    else:
        original_indices_test = torch.load('original_indices_test')

    indices = original_indices_test[:num_samples_test]
    anti_indices = original_indices_test[num_samples_test:num_samples_test + measurement_number]
    attacker_test_images = test_images[indices]
    attacker_test_labels = test_labels[indices]
    measurement_test_images = test_images[anti_indices]
    measurement_test_labels = test_labels[anti_indices]

    # Build shadow and measurement sets
    shadow_images = torch.cat([attacker_train_images, attacker_test_images])
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels])
    measurement_images = torch.cat([measurement_train_images, measurement_test_images])
    measurement_ref = np.array([0] * len(measurement_train_images) + [1] * len(measurement_test_images))
    measurement_labels = torch.cat([measurement_train_labels, measurement_test_labels])

    print("Measurement Sample Size:", len(measurement_images))

    # ------------------- Run MIA -------------------
    scores = run_over_MIA(
        target_model,
        measurement_images,
        shadow_images,
        shadow_labels,
        num_shadow_models,
        epochs_shadow_model,
        lr_shadow_model,
        epochs_attack_model,
        lr_attack_model,
        attack_hidden_size,
        device
    )

    # ------------------- ROC & reporting -------------------
    fpr, tpr, _ = roc_curve(measurement_ref, scores)
    roc_auc = auc(fpr, tpr)
    print("--------------")
    print(f"AUC: {roc_auc}")
    print("-------------")

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('results/ROC_Shadow.png')

    # Save hyperparameters/config
    filename = "results/Shadow_Attack.txt"
    hyperparams = {
        "input_shape": input_shape,
        "channel": channel,
        "num_classes": num_classes,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "epochs": epochs,
        "lr": lr,
        "perc": perc,
        "perc_test": perc_test,
        "measurement_number": measurement_number,
        "num_shadow_models": num_shadow_models,
        "lr_shadow_model": lr_shadow_model,
        "epochs_shadow_model": epochs_shadow_model,
        "lr_attack_model": lr_attack_model,
        "epochs_attack_model": epochs_attack_model,
        "attack_hidden_size": attack_hidden_size,
        "device": str(device),
        "net": net_name,
        "classes": classes,
        "weight_path": weight_path,
        "batch_size": batch_size
    }
    os.makedirs("results", exist_ok=True)
    with open(filename, 'w') as f:
        for param_name, param_value in hyperparams.items():
            f.write(f"{param_name}: {param_value}\n")


if __name__ == "__main__":
    main()
