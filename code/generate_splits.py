import pandas as pd
import numpy as np
import os
from collections import Counter
from torch.utils.data import Subset

def generate_and_save_balanced_split(trainset, dataset_name, forget_per_class, num_classes, seeds, output_csv_path):
    """
    Generates balanced per-class splits for dataset and saves to CSV.
    Columns: index, label, split, seed, dataset, forget_per_class
    """
    all_rows = []
    # FIXED: Handle both .targets (CIFAR) and .labels (MUCAC) attributes
    if hasattr(trainset, 'targets'):
        targets = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        targets = np.array(trainset.labels)
    else:
        raise AttributeError(f"Dataset {dataset_name} has neither 'targets' nor 'labels' attribute")

    for seed in seeds:
        np.random.seed(seed)
        label_counter = []

        print(f"\n	1 Seed {seed}:")

        for class_label in range(num_classes):
            class_indices = np.where(targets == class_label)[0]
            class_indices = np.random.permutation(class_indices)

            forget_class_indices = class_indices[:forget_per_class]
            retain_class_indices = class_indices[forget_per_class:]

            label_counter.extend([class_label] * len(retain_class_indices))

            for idx in forget_class_indices:
                all_rows.append({
                    "index": idx,
                    "label": class_label,
                    "split": "forget",
                    "seed": seed,
                    "dataset": dataset_name,
                    "forget_per_class": forget_per_class
                })

            for idx in retain_class_indices:
                all_rows.append({
                    "index": idx,
                    "label": class_label,
                    "split": "retain",
                    "seed": seed,
                    "dataset": dataset_name,
                    "forget_per_class": forget_per_class
                })

        retain_dist = Counter(label_counter)
        for c in range(num_classes):
            print(f"Class {c}: {retain_dist[c]} retain")

    # Save CSV
    df = pd.DataFrame(all_rows)[["index", "label", "split", "seed", "dataset", "forget_per_class"]]
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True) if os.path.dirname(output_csv_path) else None
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved split results to {output_csv_path}")


dataset_name = "Cifar10"
num_classes = 10
forget_per_class = 500
seeds = list(range(11, 51)) 
output_csv_path = "splits/split_indices_Cifar10_50.csv"

generate_and_save_balanced_split(
    trainset=trainset,
    dataset_name=dataset_name,
    forget_per_class=forget_per_class,
    num_classes=num_classes,
    seeds=seeds,
    output_csv_path=output_csv_path
)
