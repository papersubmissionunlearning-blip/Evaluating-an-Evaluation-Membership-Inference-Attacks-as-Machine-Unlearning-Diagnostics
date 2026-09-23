import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Times New Roman'

# Define file paths
# files = {
#     "CIFAR-10 AllCNN": r"C:/Temp/Unlearning/Results/Cifar 10 AllCNN/compiled_results_MIAU.csv",
#     "CIFAR-10 ResNet-18": r"C:/Temp/Unlearning/Results/Cifar 10 Resnet/compiled_results_MIAU.csv",
#     "CIFAR-20 AllCNN": r"C:/Temp/Unlearning/Results/Cifar 20 AllCNN/compiled_results_MIAU.csv",
#     "CIFAR-20 ResNet-18": r"C:/Temp/Unlearning/Results/Cifar 20 Resnet/compiled_results_MIAU.csv",
#     "CIFAR-10 ViT": r"C:/Temp/Unlearning/Results/Cifar 10 ViT/compiled_results_MIAU.csv",
#     "MNIST ResNet-18": r"C:/Temp/Unlearning/Results/MNIST Resnet/compiled_results_MIAU.csv",
#     "MNIST AllCNN": r"C:/Temp/Unlearning/Results/MNIST AllCNN/compiled_results_MIAU.csv",
#     "MUCAC ResNet-18": r"C:/Temp/Unlearning/Results/MUCAC Resnet/compiled_results_MIAU.csv"
# }
files = {
    #"CIFAR10 ResNet Underfitted": r"C:/Temp/Unlearning/Results/Underfitted/compiled_results_MIAU.csv",
    "CIFAR10 ResNet Overfitted": r"C:/Temp/Unlearning/Results/Overfitted/compiled_results_MIAU.csv"
}
mia_columns = [
    "Forget vs Retain Membership Inference Attack (MIA)",
    "Forget vs Test Membership Inference Attack (MIA)",
    "Test vs Retain Membership Inference Attack (MIA)"
]

plot_data = []

for dataset_name, file_path in files.items():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if 'unlearning' not in df.columns:
        print(f"Missing 'unlearning' column in {dataset_name}")
        continue

    for col in mia_columns:
        if col not in df.columns:
            print(f"Missing column '{col}' in {dataset_name}")
            continue

        for phase, label in [('baseline', 'Before Unlearning'), ('retrain', 'After Unlearning')]:
            subset = df[df['unlearning'] == phase][col].dropna()
            for val in subset:
                plot_data.append({
                    "Phase": label,
                    "Metric": col.replace(" Membership Inference Attack (MIA)", ""),
                    "Score": val
                })

df_plot = pd.DataFrame(plot_data)

print(df_plot.groupby(["Metric", "Phase"]).size().reset_index(name="Count"))

custom_palette = {
    "Before Unlearning": "#e74c3c",  
    "After Unlearning": "#f1948a"  
}

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x="Metric", y="Score", hue="Phase", palette=custom_palette, fliersize=3)
sns.stripplot(data=df_plot, x="Metric", y="Score", hue="Phase", dodge=True, jitter=True, alpha=0.3, color=".3", legend=False)

plt.title("MIA Score Distribution: Baseline vs Retrain", fontname='Times New Roman')
plt.ylabel("MIA Score", fontname='Times New Roman')
plt.xlabel("MIA Comparison", fontname='Times New Roman')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:/Temp/Unlearning/figure_baseline_vs_retrain_overfitted.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()
