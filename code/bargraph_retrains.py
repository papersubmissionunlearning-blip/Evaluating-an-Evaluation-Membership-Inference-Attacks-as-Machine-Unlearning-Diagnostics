


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Times New Roman'

files = {
    #!!! When swaping MUCAC Be careful only swap it 
    
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
methods = ['retrain25', 'retrain50', 'retrain75']
label_map = {'retrain25': '25%', 'retrain50': '50%', 'retrain75': '75%'}

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

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid", font_scale=1.2)

ax = sns.barplot(
    data=df_plot,
    x="Retrain %",
    y="Average MIAU",
    hue="Dataset",
    ci=None,
    capsize=0.1
)

retrain_levels = ["25%", "50%", "75%"]
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
            ax.errorbar(x_center, avg, yerr=std, fmt='none', c='black', capsize=4, linewidth=1.5)

plt.title("Average MIAU Score ± Std Across Seeds", fontname='Times New Roman')
plt.ylabel("Average MIAU", fontname='Times New Roman')
plt.xlabel("Retraining Level", fontname='Times New Roman')
plt.legend(title="Experiment", bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=3)
plt.tight_layout()

plt.savefig(r"C:/Temp/Unlearning/figure_bargraph_retrains_nongeneralized.pdf", dpi=300, bbox_inches='tight')
plt.show()




