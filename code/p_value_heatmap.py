

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Figures are drawn at the width they are printed at (NeurIPS \linewidth = 5.5in);
# NeurIPS does not accept Type 3 fonts, so fonts are embedded as TrueType (42)
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.linewidth': 0.6, 'axes.titleweight': 'bold',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
OUTPUT_DIR = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"

standard_folders = {
        "CIFAR-10 AllCNN": "Results_AllCNN_Cifar10",
       "CIFAR-10 ResNet-18": "Results_ResNet18_Cifar10",
       "CIFAR-10 ResNet-18 Saliency": "Results_ResNet18_Cifar10_AdvancedMIA_Xgboost",
       "CIFAR-10 ViT": "Results_ViT_Cifar10",
       "CIFAR-20 AllCNN": "Results_AllCNN_Cifar20",
       "CIFAR-20 AllCNN Full-Class": "Results_AllCNN_Cifar20_FullClass",
       "CIFAR-20 AllCNN Sub-Class": "Results_AllCNN_Cifar20_SubClass",
       "CIFAR-20 ResNet-18": "Results_ResNet18_Cifar20",
       "MNIST AllCNN": "Results_AllCNN_Mnist",
       "MNIST ResNet-18": "Results_ResNet18_Mnist",
       "MUCAC ResNet-18": "Results_ResNet18_MUCAC",
}

nongeneralized_folders = {
  "CIFAR-10 ResNet-18 Overfitted": "Results_ResNet18_Cifar10_Overfitted",
       "CIFAR-10 ResNet-18 Underfitted": "Results_ResNet18_Cifar10_Underfitted",
}

def csv_paths(folders, csv_name):
    return {name: os.path.join(RESULTS_ROOT, folder, csv_name) for name, folder in folders.items()}

# (output name, files, figure width in inches)
variants = [
    ("figure_p_value_heatmap",
     csv_paths(standard_folders, "compiled_results_MIAU.csv"), 3), #2.64  6
    ("figure_p_value_heatmap_nongeneralized",
     csv_paths(nongeneralized_folders, "compiled_results_MIAU.csv"), 5.5),
    ("figure_p_value_heatmap_0.5_alpha",
     csv_paths(standard_folders, "compiled_results_ablation_MIAU.csv"), 4.4),
]


comparisons = [
    ("retrain50", "retrain25"),
    ("retrain75", "retrain25"),
    ("retrain75", "retrain50")
]


def plot_pvalue_heatmap(files, output_name, width):
    pval_dict = {}

    for name, path in files.items():
        try:
            df = pd.read_csv(path)

            methods = ['retrain25', 'retrain50', 'retrain75']
            df_filtered = df[df['unlearning'].isin(methods)]
            df_pivot = df_filtered.pivot(index='seed', columns='unlearning', values='MIAU')

            if not all(method in df_pivot.columns for method in methods):
                continue

            result = {}
            for high, low in comparisons:
                pval = ttest_rel(df_pivot[high], df_pivot[low], alternative='greater').pvalue
                result[f"{high} > {low}"] = pval

            pval_dict[name] = result

        except Exception as e:
            print(f"Error processing {name}: {e}")

    df_pvals = pd.DataFrame.from_dict(pval_dict, orient='index')

    fig = plt.figure(figsize=(width, 0.6 + len(df_pvals) * 0.2))
    ax = sns.heatmap(df_pvals, annot=True, fmt=".4f", cmap="Reds", annot_kws={'fontsize': 6},
                     cbar_kws={'label': 'p-value'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6, width=0.5)
    cbar.ax.yaxis.label.set_size(6.5)
    if width < 4:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    plt.title("One-sided p-values for MIAU comparisons")
    plt.xlabel("Comparison")
    plt.ylabel("Dataset")
    plt.tight_layout(pad=0.3)
    save_path = os.path.join(OUTPUT_DIR, output_name + ".pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for output_name, files, width in variants:
        plot_pvalue_heatmap(files, output_name, width)
