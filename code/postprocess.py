
#################### ------------- 1 Advanced Attack Fix ------------#############################

# import os
# import sys
# import shutil

# import pandas as pd

# sys.path.insert(0, r"C:\Temp\Unlearning\Code Appendix\source\code")
# from calculate_miau_per_seed import calculate_miau_for_csv

# ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
# FILES = [
#     os.path.join(ROOT, f"Results_ResNet18_Cifar10_AdvancedMIA_{attack}", "compiled_results_MIAU.csv")
#     for attack in ("Lira", "Shadow", "Quantile")
# ]
# FLIP_COLS = [
#     "Forget vs Test Membership Inference Attack (MIA)",
#     "Train vs Test Membership Inference Attack (MIA)",
# ]

# for path in FILES:
#     # Keep the unflipped original once and always flip from it, so re-running
#     # this cell can never flip a column twice.
#     backup = path.replace(".csv", "_before_auc_flip.csv")
#     if not os.path.exists(backup):
#         shutil.copyfile(path, backup)

#     before = pd.read_csv(backup)
#     df = before.copy()
#     for col in FLIP_COLS:
#         df[col] = 1 - df[col]
#     df.to_csv(path, index=False)

#     # Recomputes f_i, MUS_i and MIAU from the flipped columns and overwrites them in place
#     after = calculate_miau_for_csv(path, path)

#     summary = pd.DataFrame({
#         "MIAU before": before.groupby("unlearning")["MIAU"].mean(),
#         "MIAU after": after.groupby("unlearning")["MIAU"].mean(),
#     }).round(2)
#     print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
#     print(summary)



#################### ------------- 2 Resnet Duo Fix------------#############################
# import pandas as pd
# import sys
# sys.path.insert(0, r"C:\Temp\Unlearning\Code Appendix\source\code")
# from calculate_miau_per_seed import calculate_miau_for_csv
# path = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10\compiled_results_MIAU.csv"

# df = pd.read_csv(path)

# # Find rows where dataset is 'do' (incorrectly parsed _do methods)
# mask = df['dataset'] == 'do'

# # For these rows:
# # 1. Append '_do' to the unlearning method name
# # 2. Split the 'model' column (which contains 'cifar10_resnet18') into dataset and model
# for idx in df[mask].index:
#     original_unlearning = df.loc[idx, 'unlearning']
#     combined_model = df.loc[idx, 'model']  # e.g., 'cifar10_resnet18'
    
#     # Split 'cifar10_resnet18' into 'cifar10' and 'resnet18'
#     parts = combined_model.split('_', 1)  # Split on first underscore only
#     new_dataset = parts[0]  # 'cifar10'
#     new_model = parts[1] if len(parts) > 1 else combined_model  # 'resnet18'
    
#     # Update the row
#     df.loc[idx, 'unlearning'] = f"{original_unlearning}_do"
#     df.loc[idx, 'dataset'] = new_dataset
#     df.loc[idx, 'model'] = new_model

# # Save the fixed file
# df.to_csv(path, index=False)
# calculate_miau_for_csv(path, path)
# check = pd.read_csv(path)
# print(check.groupby("unlearning")["MIAU"].agg(["count", "mean"]))
# print(f"Fixed {mask.sum()} rows with _do methods")

#################### ------------- 3 SSD Tuning Fix ------------#############################



import pandas as pd
files = {
    # "CIFAR10_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar10\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10\compiled_results_MIAU.csv",
    "CIFAR10_ResNet18_Overfitted": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_Overfitted\compiled_results_MIAU.csv",
    "CIFAR10_ResNet18_Underfitted": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_Underfitted\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Shadow": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Shadow\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Quantile": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Quantile\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Lira": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Lira\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Xgboost": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Xgboost\compiled_results_MIAU.csv",
    # "CIFAR10_ViT": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ViT_Cifar10\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN_FullClass": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20_FullClass\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN_SubClass": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20_SubClass\compiled_results_MIAU.csv",
    # "CIFAR20_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar20\compiled_results_MIAU.csv",
    # "MNIST_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Mnist\compiled_results_MIAU.csv",
    # "MNIST_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Mnist\compiled_results_MIAU.csv",
    # "MUCAC_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_MUCAC\compiled_results_MIAU.csv",
}
for name, path in files.items():
    try:
        df = pd.read_csv(path)
        df['unlearning'] = df['unlearning'].replace('ssdtuning', 'ssd')
        df.to_csv(path, index=False)
        print(f"Replaced ssdtuning -> ssd in {name}")
    except Exception as e:
        print(f"Error processing {name}: {e}")



#################### ------------- 4  Swap 25/75------------#############################

import pandas as pd

# Dictionary of result files
files = {
    # "CIFAR10_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar10\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10\compiled_results_MIAU.csv",
    "CIFAR10_ResNet18_Overfitted": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_Overfitted\compiled_results_MIAU.csv",
    "CIFAR10_ResNet18_Underfitted": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_Underfitted\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Shadow": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Shadow\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Quantile": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Quantile\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Lira": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Lira\compiled_results_MIAU.csv",
    # "CIFAR10_ResNet18_AdvancedMIA_Xgboost": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar10_AdvancedMIA_Xgboost\compiled_results_MIAU.csv",
    # "CIFAR10_ViT": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ViT_Cifar10\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN_FullClass": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20_FullClass\compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN_SubClass": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Cifar20_SubClass\compiled_results_MIAU.csv",
    # "CIFAR20_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Cifar20\compiled_results_MIAU.csv",
    # "MNIST_AllCNN": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_AllCNN_Mnist\compiled_results_MIAU.csv",
    # "MNIST_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_Mnist\compiled_results_MIAU.csv",
    # "MUCAC_ResNet18": r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\Results_ResNet18_MUCAC\compiled_results_MIAU.csv",
}
for name, path in files.items():
    try:
        df = pd.read_csv(path)

        # Swap labels
        df['unlearning'] = df['unlearning'].replace({
            'retrain25': 'TEMP_SWAP_TAG',
            'retrain75': 'retrain25'
        })
        df['unlearning'] = df['unlearning'].replace({
            'TEMP_SWAP_TAG': 'retrain75'
        })

        # Overwrite file (optional: save to new file)
        df.to_csv(path, index=False)
        print(f"Swapped retrain25 <-> retrain75 in {name}")
        
    except Exception as e:
        print(f"Error processing {name}: {e}")
        
        
        
#################### ------------- 5 Ablation Study ------------#############################
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Temp\Unlearning\Code Appendix\source\code")
from calculate_miau_per_seed import compute_fi, compute_musi

ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
FOLDERS = [
    # "Results_AllCNN_Cifar10",
    # "Results_ResNet18_Cifar10",
    "Results_ResNet18_Cifar10_Overfitted",
    "Results_ResNet18_Cifar10_Underfitted",
    # "Results_ResNet18_Cifar10_AdvancedMIA_Shadow",
    # "Results_ResNet18_Cifar10_AdvancedMIA_Quantile",
    # "Results_ResNet18_Cifar10_AdvancedMIA_Lira",
    # "Results_ResNet18_Cifar10_AdvancedMIA_Xgboost",
    # "Results_ViT_Cifar10",
    # "Results_AllCNN_Cifar20",
    # "Results_AllCNN_Cifar20_FullClass",
    # "Results_AllCNN_Cifar20_SubClass",
    # "Results_ResNet18_Cifar20",
    # "Results_AllCNN_Mnist",
    # "Results_ResNet18_Mnist",
    # "Results_ResNet18_MUCAC",
]

ABLATION_COLS = [
    "Forget vs Retain Membership Inference Attack (MIA)",
    "Forget vs Test Membership Inference Attack (MIA)",
]
DROPPED_COL = "Test vs Retain Membership Inference Attack (MIA)"
KEYS = ["seed", "dataset", "model"]


def ablation_miau(path):
    df = pd.read_csv(path)
    full_miau = df["MIAU"].copy() if "MIAU" in df.columns else None

    # The raw Test vs Retain AUC stays; only its derived scores are removed
    df = df.drop(columns=[f"f_i ({DROPPED_COL})", f"MUS_i ({DROPPED_COL})"], errors="ignore")

    keys = pd.MultiIndex.from_frame(df[KEYS])
    refs = {
        name: df[df["unlearning"] == name].drop_duplicates(KEYS).set_index(KEYS)
        for name in ("baseline", "retrain")
    }

    for col in ABLATION_COLS:
        B = refs["baseline"][col].reindex(keys).to_numpy()
        R = refs["retrain"][col].reindex(keys).to_numpy()
        M = df[col].to_numpy()
        f_i = np.array([
            np.nan if pd.isna(b) or pd.isna(r) or pd.isna(m) else compute_fi(b, r, m)
            for b, r, m in zip(B, R, M)
        ], dtype=float)
        df[f"f_i ({col})"] = f_i
        df[f"MUS_i ({col})"] = compute_musi(f_i)

    df["MIAU"] = df[[f"MUS_i ({c})" for c in ABLATION_COLS]].mean(axis=1)

    base, ext = os.path.splitext(path)
    if base.lower().endswith("_miau"):
        base = base[: -len("_miau")]
    out_path = f"{base}_ablation_MIAU{ext}"
    df.to_csv(out_path, index=False)

    summary = pd.DataFrame({"MIAU (ablation)": df.groupby("unlearning")["MIAU"].mean()})
    if full_miau is not None:
        summary.insert(0, "MIAU (full)", full_miau.groupby(df["unlearning"]).mean())
    print(f"\n=== {os.path.basename(os.path.dirname(path))} -> {os.path.basename(out_path)}")
    print(summary.round(2))


for folder in FOLDERS:
    path = os.path.join(ROOT, folder, "compiled_results_MIAU.csv")
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        continue
    ablation_miau(path)