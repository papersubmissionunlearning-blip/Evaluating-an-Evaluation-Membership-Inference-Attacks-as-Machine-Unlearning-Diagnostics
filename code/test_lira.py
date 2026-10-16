import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
import math
import matplotlib.pyplot as plt 
import os
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from lira.shadow import *
from lira.lira import *
from lira.model import *
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
from collections import Counter


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
  
#-----------------------------------------------------------------------------------
input_shape = (3, 32, 32)
channel = 3
num_classes=10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
perc=0.0   # amount of actual training data available to the attacker
perc_test=0.20    # amount of testing data available to the attacker ( similar distribution to training data)
meausurement_number=30   # number of target samples to be measured from each training and non-training data
num_shadow_models=250
lr_shadow_model=1e-3
epochs_shadow_model=20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------------------------------------------------------------------------


if not os.path.exists('results'):
    os.makedirs('results')

# Load CIFAR-10 dataset
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#    transforms.Lambda(lambda x: x.view(3, 32, 32))
#])

#train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


#batch_size = 128 * 2
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
img_size = 32
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



#target_model=CNN(channel, num_classes).to(device)
#if not os.path.exists('target_model.pth'):
#    print(f"Training Target Model on CIFAR-10 on Epochs: {epochs}")
#    train_model_with_loader(target_model, train_loader, epochs, lr,device)
#    torch.save(target_model.state_dict(), 'target_model.pth')
#
#else:
#    print("Loading trained Target Model")
#    target_model.load_state_dict(torch.load('target_model.pth'))
#    target_model.to(device)
net_name = "ResNet18"  # must match torchvision.models attribute name
classes = 10
batch_size = 256
# default weight path in current working directory
current_dir = os.getcwd()
weight_path = os.path.join(current_dir, "checkpoint", "ResNet18", "ResNet18-Cifar10-20-best.pth")

target_model = getattr(models, net_name)(num_classes=classes).to(device)

    # Load or train
if weight_path and os.path.exists(weight_path):
  print(f"Loading weights from {weight_path}")
  state = torch.load(weight_path, map_location=device)
  target_model.load_state_dict(state)

# default_model_path = r"C:\Users\aliev\checkpoint\ResNet18\Monday_02_December_2024_02h_20m_31s\ResNet18-Cifar10-20-best.pth"
# target_model = getattr(models, "ResNet18")(num_classes=10)
# target_model.load_state_dict(torch.load(default_model_path))
# target_model = target_model.cuda()


# Calculate training and test accuracy
#train_accuracy = calculate_accuracy(target_model, train_loader, device)
#test_accuracy = calculate_accuracy(target_model, test_loader, device)
#print(f"Training Accuracy: {train_accuracy:.2f}%")
#print(f"Test Accuracy: {test_accuracy:.2f}%")





num_samples_train = int(perc*len(train_loader.dataset))
num_samples_test=int(perc_test*len(test_loader.dataset))
print("----------------------------------")
print(f"Attackers knowledge:")
print(f"Training Dataset Info: {num_samples_train}/{len(train_loader.dataset)} = {num_samples_train/len(train_loader.dataset)*100}%")
print(f"Testing Dataset Info: {num_samples_test}/{len(test_loader.dataset)} = {num_samples_test/len(test_loader.dataset)*100}%")
print("----------------------------------")


train_images=[]
train_labels=[]
for images, _, labels in train_loader:
    train_images.append(images)
    train_labels.append(labels)
train_images=torch.cat(train_images)
train_labels=torch.cat(train_labels)

if not os.path.exists('original_indices'):
    original_indices=torch.randperm(len(train_images))
    torch.save(original_indices,'original_indices')
else:
    original_indices=torch.load('original_indices')

indices = original_indices[:num_samples_train]
anti_indices = original_indices[num_samples_train:num_samples_train+meausurement_number]
attacker_train_images = train_images[indices]
attacker_train_labels = train_labels[indices]

measurement_train_images = train_images[anti_indices]
measurement_train_labels = train_labels[anti_indices]


test_images=[]
test_labels=[]
for images, _, labels in test_loader:
    test_images.append(images)
    test_labels.append(labels)
test_images=torch.cat(test_images)
test_labels=torch.cat(test_labels)

if not os.path.exists('original_indices_test'):
    original_indices_test=torch.randperm(len(test_images))
    torch.save(original_indices_test,'original_indices_test')
else:
    original_indices_test=torch.load('original_indices_test')


indices = original_indices_test[:num_samples_test]
anti_indices = original_indices_test[num_samples_test:num_samples_test+meausurement_number]


attacker_test_images = test_images[indices]
attacker_test_labels = test_labels[indices]

measurement_test_images = test_images[anti_indices]
measurement_test_labels = test_labels[anti_indices]

shadow_images=torch.cat([attacker_train_images,attacker_test_images])
shadow_labels=torch.cat([attacker_train_labels,attacker_test_labels])


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array([0]*len(measurement_train_images)+[1]*len(measurement_test_images))
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])

print("Measurement Sample Size:",len(measurement_images))



# --------LIRA-----------------------------------------------------------------
# shadow zone
in_mean, in_std, out_mean, out_std = estimate_loss_distributions(measurement_images, 
                                                                 measurement_labels, 
                                                                 shadow_images, 
                                                                 shadow_labels,
                                                                 num_shadow_models=num_shadow_models, 
                                                                 epochs=epochs_shadow_model, 
                                                                 lr=lr_shadow_model)
# attack zone
scores = run_over_MIA(target_model, 
                          measurement_images, 
                          measurement_labels, 
                          in_mean, 
                          in_std, 
                          out_mean, 
                          out_std)

#-----------------------------------------------------------------------------------
np.savetxt("scores.txt", scores, fmt='%.6f')
np.savetxt("measurement_ref.txt", measurement_ref, fmt='%.6f')

#tpr, fpr, roc = roc_curve(measurement_ref, scores)
fpr, tpr, thresholds = roc_curve(measurement_ref, scores)

print("--------------")
print(f"AUC: {auc(fpr, tpr):.4f}")

print("-------------")
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/ROC_LIRA Attack.png')




filename = "results/LIRA.txt"

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
    "measurement_number": meausurement_number,
    "num_shadow_models": num_shadow_models,
    "lr_shadow_model": lr_shadow_model,
    "epochs_shadow_model": epochs_shadow_model,
    "device": str(device)
}


os.makedirs("results", exist_ok=True)

with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")