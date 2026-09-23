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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.utilities.data import to_onehot
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import warnings
warnings.filterwarnings("ignore")
from quantile.model import *
from quantile.quantile_attack import *
import models

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
meausurement_number=10 # number of target samples to be measured from each training and non-training data
auxillary_bs=32
n_quantile=100
low_quantile=0.01
high_quantile=0.99
use_logscale=False
use_gaussian=False
batch_size=32
quantile_model_epochs=30
quantile_model_lr=1e-3
alpha=0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------------------------------------------------------------------------

if not os.path.exists('results'):
    os.makedirs('results')

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x: x.view(3, 32, 32))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# target_model=CNN(channel, num_classes).to(device)
# if not os.path.exists('target_model.pth'):
#     print(f"Training Target Model on CIFAR-10 on Epochs: {epochs}")
#     train_model_with_loader(target_model, train_loader, epochs, lr,device)
#     torch.save(target_model.state_dict(), 'target_model.pth')

# else:
#     print("Loading trained Target Model")
#     target_model.load_state_dict(torch.load('target_model.pth'))
#     target_model.to(device)
#default_model_path = r"C:\Users\aliev\checkpoint\ResNet18\Monday_02_December_2024_02h_20m_31s\ResNet18-Cifar10-20-best.pth"
#target_model = getattr(models, "ResNet18")(num_classes=10)
#target_model.load_state_dict(torch.load(default_model_path))
#target_model = target_model.cuda()

net_name = "ResNet18"
classes = 10
batch_size = 256
current_dir = os.getcwd()
weight_path = os.path.join(current_dir, "checkpoint", "ResNet18", "ResNet18-Cifar10-20-best.pth")

target_model = getattr(models, net_name)(num_classes=classes).to(device)
if weight_path and os.path.exists(weight_path):
  print(f"Loading weights from {weight_path}")
  state = torch.load(weight_path, map_location=device)
  target_model.load_state_dict(state)
  
# Calculate training and test accuracy
# = calculate_accuracy(target_model, train_loader, device)
#test_accuracy = calculate_accuracy(target_model, test_loader,device)
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
for images, labels in train_loader:
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
for images, labels in test_loader:
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

auxillary_dataset = TensorDataset(shadow_images, shadow_labels)
axuillary_loader = DataLoader(auxillary_dataset, batch_size=auxillary_bs, shuffle=True)


target_model=target_model.to('cpu')
scores= mia_attack(target_model, 
                    measurement_images, 
                    measurement_labels, 
                    axuillary_loader,
                    n_quantile=n_quantile,
                    low_quantile=low_quantile,
                    high_quantile=high_quantile,
                    use_logscale=use_logscale,
                    use_gaussian=use_gaussian,
                    batch_size=batch_size,
                    num_epochs=quantile_model_epochs,
                    learning_rate=quantile_model_lr,
                    alpha=alpha,
                    device=torch.device('cuda')
                    )

fpr, tpr, roc = roc_curve(measurement_ref, scores)
print("--------------")
print(f'AUC: {auc(fpr, tpr)}  |')
print("-------------")
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/ROC_Quantile Attack.png')



filename = "results/quantile_regression.txt"


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
    "auxillary_bs": auxillary_bs,
    "n_quantile": n_quantile,
    "low_quantile": low_quantile,
    "high_quantile": high_quantile,
    "use_logscale": use_logscale,
    "use_gaussian": use_gaussian,
    "batch_size": batch_size,
    "quantile_model_epochs": quantile_model_epochs,
    "quantile_model_lr": quantile_model_lr,
    "alpha": alpha,
    "device": str(device)
}


os.makedirs("results", exist_ok=True)

with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")