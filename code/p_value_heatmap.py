

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
plt.rcParams['font.family'] = 'Times New Roman'

# Define dictionary of files
files = {    
    # "CIFAR10_AllCNN": r"C:/Temp/Unlearning/Results/Cifar 10 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR10_ResNet": r"C:/Temp/Unlearning/Results/Cifar 10 Resnet/compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN": r"C:/Temp/Unlearning/Results/Cifar 20 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR20_ResNet": r"C:/Temp/Unlearning/Results/Cifar 20 Resnet/compiled_results_MIAU.csv",
    # "CIFAR10_ViT": r"C:/Temp/Unlearning/Results/Cifar 10 ViT/compiled_results_MIAU.csv",
    # "MNIST_ResNet": r"C:/Temp/Unlearning/Results/MNIST Resnet/compiled_results_MIAU.csv",
    # "MNIST_AllCNN": r"C:/Temp/Unlearning/Results/MNIST AllCNN/compiled_results_MIAU.csv",
    # "MUCAC_ResNet": r"C:/Temp/Unlearning/Results/MUCAC Resnet/compiled_results_MIAU.csv"
    
     "CIFAR10_ResNet_Underfitted": r"C:/Temp/Unlearning/Results/Underfitted/compiled_results_MIAU.csv",
    "CIFAR10_ResNet_Overfitted": r"C:/Temp/Unlearning/Results/Overfitted/compiled_results_MIAU.csv"
}


comparisons = [
    ("retrain50", "retrain25"),
    ("retrain75", "retrain25"),
    ("retrain75", "retrain50")
]

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

plt.figure(figsize=(10, len(df_pvals) * 0.6))
sns.heatmap(df_pvals, annot=True, fmt=".4f", cmap="Reds", cbar_kws={'label': 'p-value'})
plt.title("One-sided p-values for MIAU comparisons", fontname='Times New Roman')
plt.xlabel("Comparison", fontname='Times New Roman')
plt.ylabel("Dataset", fontname='Times New Roman')
plt.tight_layout()
plt.savefig(r"C:/Temp/Unlearning/figure_p_value_heatmap_nongeneralized.pdf", dpi=300, bbox_inches='tight')

plt.show()

