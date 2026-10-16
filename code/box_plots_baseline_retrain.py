import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Figures are drawn at the width they are printed at (NeurIPS \linewidth = 5.5in);
# NeurIPS does not accept Type 3 fonts, so fonts are embedded as TrueType (42)
sns.set_theme(style="whitegrid", font="serif", rc={
    'font.serif': ['Times New Roman'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4,
    'axes.titleweight': 'bold',
    # whitegrid defaults text to dark grey and spines to light grey; match the other figures
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
OUTPUT_DIR = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"

standard_files = {
        "CIFAR-10 AllCNN": os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar10", "compiled_results_MIAU.csv"),
       "CIFAR-10 ResNet-18": os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10", "compiled_results_MIAU.csv"),
       "CIFAR-10 ViT": os.path.join(RESULTS_ROOT, "Results_ViT_Cifar10", "compiled_results_MIAU.csv"),
       "CIFAR-20 AllCNN": os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar20", "compiled_results_MIAU.csv"),
       "CIFAR-20 ResNet-18": os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar20", "compiled_results_MIAU.csv"),
       "MNIST AllCNN": os.path.join(RESULTS_ROOT, "Results_AllCNN_Mnist", "compiled_results_MIAU.csv"),
       "MNIST ResNet-18": os.path.join(RESULTS_ROOT, "Results_ResNet18_Mnist", "compiled_results_MIAU.csv"),
       "MUCAC ResNet-18": os.path.join(RESULTS_ROOT, "Results_ResNet18_MUCAC", "compiled_results_MIAU.csv"),
}

overfitted_files = {
  "CIFAR-10 ResNet-18 Overfitted": os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_Overfitted", "compiled_results_MIAU.csv"),
}

underfitted_files = {
       "CIFAR-10 ResNet-18 Underfitted": os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_Underfitted", "compiled_results_MIAU.csv"),
}

# (output name, files, figure size in inches)
variants = [
    ("figure_baseline_vs_retrain", standard_files, (2.64, 1.9)),
    ("figure_baseline_vs_retrain_overfitted", overfitted_files, (3.85, 2.1)),
    ("figure_baseline_vs_retrain_underfitted", underfitted_files, (3.85, 2.1)),
]


mia_columns = [
    "Forget vs Retain Membership Inference Attack (MIA)",
    "Forget vs Test Membership Inference Attack (MIA)",
    "Test vs Retain Membership Inference Attack (MIA)"
]

custom_palette = {
    "Before Unlearning": "#e74c3c",  
    "After Unlearning": "#f1948a"  
}


def plot_boxplots(files, output_name, figsize):
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

    print(output_name)
    print(df_plot.groupby(["Metric", "Phase"]).size().reset_index(name="Count"))

    fig = plt.figure(figsize=figsize)
    sns.boxplot(data=df_plot, x="Metric", y="Score", hue="Phase", palette=custom_palette,
                fliersize=1.5, linewidth=0.6)
    sns.stripplot(data=df_plot, x="Metric", y="Score", hue="Phase", dodge=True, jitter=True,
                  alpha=0.3, color=".3", size=1.5, legend=False)

    plt.title("MIA Score Distribution: Baseline vs Retrain")
    plt.ylabel("MIA Score")
    plt.xlabel("MIA Comparison", labelpad=10)
    plt.xticks(rotation=0)
    plt.legend(title="Phase", title_fontsize=6, frameon=True, handlelength=1.0)

    plt.tight_layout(pad=0.3)
    save_path = os.path.join(OUTPUT_DIR, output_name + ".pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for output_name, files, figsize in variants:
        plot_boxplots(files, output_name, figsize)
