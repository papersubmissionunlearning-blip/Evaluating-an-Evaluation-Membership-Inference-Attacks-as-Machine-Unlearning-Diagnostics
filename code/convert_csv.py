"""
Convert result .txt files to a compiled CSV.

REFACTORED: Accepts folder path as command-line argument instead of hardcoded path.

Usage:
    python convert_csv.py -folder Results_Fixed/Results_ResNet18_Cifar10
    python convert_csv.py -folder /path/to/results
    
    # Or use original hardcoded path (backward compatible)
    python convert_csv.py
"""

import os
import re
import argparse
import pandas as pd

# Default folder path (original hardcoded value for backward compatibility)
DEFAULT_FOLDER_PATH = r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet 50"

# Parse arguments
parser = argparse.ArgumentParser(description="Convert result .txt files to CSV")
parser.add_argument("-folder", "-f", type=str, default=None,
                    help="Path to folder containing result .txt files")
parser.add_argument("-output", "-o", type=str, default="compiled_results.csv",
                    help="Output CSV filename (default: compiled_results.csv)")
args = parser.parse_args()

# Use provided folder or default
folder_path = args.folder if args.folder else DEFAULT_FOLDER_PATH

metrics = [
    "Test Accuracy",
    "Retain Accuracy",
    "Zero-Retain Forget \\(ZRF\\)",
    "Membership Inference Attack \\(MIA\\)",
    "Forget vs Retain Membership Inference Attack \\(MIA\\)",
    "Forget vs Test Membership Inference Attack \\(MIA\\)",
    "Test vs Retain Membership Inference Attack \\(MIA\\)",
    "Train vs Test Membership Inference Attack \\(MIA\\)",
    "Forget Set Accuracy \\(Df\\)",
    "Method Execution Time"
]

patterns = {metric: re.compile(f"{metric}:\\s+([0-9.]+)") for metric in metrics}

filename_pattern = re.compile(r"(?P<unlearning>\w+?)_(?P<dataset>\w+?)_(?P<model>\w+?)_seed_(?P<seed>\d+)\.txt")

results = []

if not os.path.exists(folder_path):
    print(f"Error: Folder not found: {folder_path}")
    exit(1)

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        match = filename_pattern.match(filename)
        if not match:
            continue  

        meta = match.groupdict() 
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        entry = {
            "Filename": filename,
            **meta  
        }

        for metric, pattern in patterns.items():
            result = pattern.search(content)
            col_name = metric.replace("\\", "")  # Clean regex slashes
            entry[col_name] = float(result.group(1)) if result else None

        results.append(entry)

if results:
    df = pd.DataFrame(results)
    output_csv = os.path.join(folder_path, args.output)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved at: {output_csv}")
    print(f"   Total results: {len(results)}")
else:
    print(f"⚠️ No matching .txt files found in: {folder_path}")
