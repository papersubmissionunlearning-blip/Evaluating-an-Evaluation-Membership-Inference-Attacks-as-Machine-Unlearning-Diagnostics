"""
Generate MUCAC split file with deterministic (sorted) ordering.
This ensures reproducibility across platforms.
"""
import pandas as pd
import numpy as np
import os
import glob
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_mucac_meta(path):
    """Load MUCAC identity and label metadata."""
    # Load identities (same path structure as datasets.py)
    with open(os.path.join(path, "CelebA-HQ-identity.txt")) as f:
        identities = dict(line.strip().split() for line in f.readlines())
    
    # Load attributes (smiling is at index 32)
    attributes_map = {"smiling": 32}
    label_map = {}
    with open(os.path.join(path, "CelebA-HQ-attribute.txt")) as f:
        lines = f.readlines()[2:]  # Skip header lines
        for line in lines:
            parts = line.strip().split()
            file_name = parts[0]
            label_map[file_name] = {attr: int(parts[idx]) for attr, idx in attributes_map.items()}
    return identities, label_map


def generate_mucac_split(root, seeds, forget_per_class, output_path):
    """
    Generate balanced per-class splits for MUCAC and save to CSV.
    
    Uses SORTED glob ordering for reproducibility.
    """
    # Identity ranges
    test_max_identity = 190           # 0-189: test set
    train_max_identity = 1970         # 190-1969: train (retain) set  
    forget_max_identity = 4855        # 1970-4854: forget set
    
    img_dir = os.path.join(root, "CelebAMask-HQ", "CelebA-HQ-img")
    print(f"Loading MUCAC from: {img_dir}")
    
    # Load metadata
    identities, label_map = load_mucac_meta(root)
    
    # Build the dataset with SORTED ordering (critical for reproducibility!)
    image_paths = []
    labels = []
    
    # Use sorted() for deterministic ordering across platforms
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        file_name = os.path.basename(img_path)
        identity = int(identities[file_name])
        smiling = label_map[file_name]["smiling"]
        if smiling == -1:
            smiling = 0
        
        # Include all training identities (both retain and forget ranges)
        # This matches how the original split was generated
        if test_max_identity <= identity < forget_max_identity:
            image_paths.append(img_path)
            labels.append(smiling)
    
    labels = np.array(labels)
    print(f"Total training samples: {len(labels)}")
    print(f"Class 0 (not smiling): {np.sum(labels == 0)}")
    print(f"Class 1 (smiling): {np.sum(labels == 1)}")
    
    # Generate splits for each seed
    all_rows = []
    num_classes = 2  # Binary classification (smiling/not smiling)
    
    for seed in seeds:
        np.random.seed(seed)
        print(f"\nSeed {seed}:")
        
        for class_label in range(num_classes):
            class_indices = np.where(labels == class_label)[0]
            class_indices = np.random.permutation(class_indices)
            
            forget_class_indices = class_indices[:forget_per_class]
            retain_class_indices = class_indices[forget_per_class:]
            
            print(f"  Class {class_label}: {len(forget_class_indices)} forget, {len(retain_class_indices)} retain")
            
            for idx in forget_class_indices:
                all_rows.append({
                    "index": idx,
                    "label": class_label,
                    "split": "forget",
                    "seed": seed,
                    "dataset": "MUCAC",
                    "forget_per_class": forget_per_class
                })
            
            for idx in retain_class_indices:
                all_rows.append({
                    "index": idx,
                    "label": class_label,
                    "split": "retain",
                    "seed": seed,
                    "dataset": "MUCAC",
                    "forget_per_class": forget_per_class
                })
    
    # Save CSV
    df = pd.DataFrame(all_rows)[["index", "label", "split", "seed", "dataset", "forget_per_class"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved split results to {output_path}")
    print(f"Total rows: {len(df)}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MUCAC split file with sorted ordering")
    parser.add_argument("--root", type=str, default="./data/MUCAC",
                        help="Root directory containing CelebAMask-HQ folder")
    parser.add_argument("--output", type=str, default="../splits/split_indices_MUCAC.csv",
                        help="Output CSV path")
    parser.add_argument("--seeds", type=str, default="1-10",
                        help="Seed range (e.g., '1-10' or '1,2,3')")
    parser.add_argument("--forget_per_class", type=int, default=527,
                        help="Number of samples to forget per class")
    
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    print("=" * 60)
    print("Generating MUCAC Split File with Sorted Ordering")
    print("=" * 60)
    print(f"Root: {args.root}")
    print(f"Seeds: {seeds}")
    print(f"Forget per class: {args.forget_per_class}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    generate_mucac_split(args.root, seeds, args.forget_per_class, args.output)
