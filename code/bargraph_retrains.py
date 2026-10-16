


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Figures are drawn at the width they are printed at (NeurIPS \linewidth = 5.5in);
# NeurIPS does not accept Type 3 fonts, so fonts are embedded as TrueType (42)
sns.set_theme(style="whitegrid", font="serif", rc={
    'font.serif': ['Times New Roman'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 5.5, 'legend.title_fontsize': 6,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4,
    'axes.titleweight': 'bold',
    # whitegrid defaults text to dark grey and spines to light grey; match the other figures
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black',
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

# (output name, files, figure size in inches, legend columns)
variants = [
    ("figure_bargraph_retrains",
     csv_paths(standard_folders, "compiled_results_MIAU.csv"), (2.64, 1.7), 2),
    ("figure_bargraph_retrains_nongeneralized",
     csv_paths(nongeneralized_folders, "compiled_results_MIAU.csv"), (3.85, 1.8), 2),
    ("figure_bargraph_retrains_0.5_alpha",
     csv_paths(standard_folders, "compiled_results_ablation_MIAU.csv"), (3.85, 1.8), 3),
]

methods = ['retrain25', 'retrain50', 'retrain75']
label_map = {'retrain25': '25%', 'retrain50': '50%', 'retrain75': '75%'}
retrain_levels = ["25%", "50%", "75%"]


def plot_bargraph(files, output_name, figsize, legend_ncol):
    plot_data = []

    for name, path in files.items():
        try:
            df = pd.read_csv(path)
            df_filtered = df[df['unlearning'].isin(methods)]
            grouped = df_filtered.groupby('unlearning')['MIAU'].agg(['mean', 'std']).reset_index()
            for _, row in grouped.iterrows():
                plot_data.append({
                    "Dataset": name,
                    "Retrain %": label_map[row['unlearning']],
                    "Average MIAU": row['mean'],
                    "Std MIAU": row['std']
                })
        except Exception as e:
            print(f"Error processing {name}: {e}")

    df_plot = pd.DataFrame(plot_data)

    fig = plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=df_plot,
        x="Retrain %",
        y="Average MIAU",
        hue="Dataset",
        order=retrain_levels,
        errorbar=None,
    )

    datasets = df_plot["Dataset"].unique()
    n_datasets = len(datasets)
    bar_width = 0.8 / n_datasets

    for j, retrain_label in enumerate(retrain_levels):
        for i, dataset in enumerate(datasets):
            subset = df_plot[(df_plot['Dataset'] == dataset) & (df_plot['Retrain %'] == retrain_label)]
            if not subset.empty:
                avg = subset['Average MIAU'].values[0]
                std = subset['Std MIAU'].values[0]
                x_center = j + (i + 0.5 - n_datasets / 2) * bar_width
                lower = max(0.0, min(std, avg))
                ax.errorbar(x_center, avg, yerr=[[lower], [std]], fmt='none', c='black',
                            capsize=1.5, linewidth=0.6, capthick=0.6)

    plt.title("Average MIAU Score ± Std Across Seeds")
    plt.ylabel("Average MIAU")
    plt.xlabel("Retraining Level")
    ax.set_ylim(bottom=0)
    ax.get_legend().remove()
    plt.tight_layout(pad=0.3)
    # Added after tight_layout so the axes keep their size; bbox_inches='tight' grows the canvas instead
    ax.legend(title="Experiment", bbox_to_anchor=(0.5, -0.42), loc='upper center',
              ncol=legend_ncol, handlelength=1.0, columnspacing=0.8, frameon=True)

    save_path = os.path.join(OUTPUT_DIR, output_name + ".pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for output_name, files, figsize, legend_ncol in variants:
        plot_bargraph(files, output_name, figsize, legend_ncol)
