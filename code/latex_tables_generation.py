import pandas as pd
import os

metric_map = {
    "Retain Accuracy": "Retain Accuracy",
     "Test Accuracy": "Test Accuracy",
    "Forget Set Accuracy (Df)": "Forget Accuracy",
    "Zero-Retain Forget (ZRF)": "ZRF Score",
    "Membership Inference Attack (MIA)": "MIA (Retain → Forget)",
    "Forget vs Retain Membership Inference Attack (MIA)": "MIA (Forget vs Retain)",
    "Forget vs Test Membership Inference Attack (MIA)": "MIA (Forget vs Test)",
    "Test vs Retain Membership Inference Attack (MIA)": "MIA (Test vs Retain)",
    "Train vs Test Membership Inference Attack (MIA)": "MIA (Train vs Test)",
    "MIAU": "MIAU",
}

non_retrain_methods = ['baseline', 'amnesiac', 'finetune', 'teacher', 'ssd']
retrain_methods = ['retrain25', 'retrain50', 'retrain75', 'retrain']

def format_mean_std(metric_name, mean, std, digits=2):
    if "Accuracy" not in metric_name and metric_name != "MIAU":
        return f"{mean * 100:.{digits}f} ± {std * 100:.{digits}f}"
    else:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

def generate_latex_table(df, dataset_name, retrain=False):
    method_list = retrain_methods if retrain else non_retrain_methods
    caption = f"Retrain Variants on {dataset_name}" if retrain else f"Experiments on {dataset_name}"
    label = f"tab:{dataset_name.lower().replace(' ', '-')}-{'retrain' if retrain else 'noretrain'}"

    rows = []
    hline_inserted = False

    for metric_key, latex_label in metric_map.items():
        if not hline_inserted and "MIA" in metric_key:
            rows.append(r"\hline")
            hline_inserted = True

        row = [latex_label]
        for method in method_list:
            try:
                mean = df.loc[method][metric_key]['mean']
                std = df.loc[method][metric_key]['std']
                row.append(format_mean_std(metric_key, mean, std, digits=4))
            except KeyError:
                row.append("–")
        rows.append(" & ".join(row) + r" \\")

    lines = [
        "\\begin{table*}[t]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{|l|{'c|' * len(method_list)}}}",
        "    \\hline",
        "    \\textbf{Metric} & " + " & ".join(f"\\textbf{{{m.capitalize()}}}" for m in method_list) + " \\\\",
        "    \\hline",
    ]
    lines += ["    " + row for row in rows]
    lines += [
        "    \\hline",
        "  \\end{tabular}",
        "\\end{table*}"
    ]

    latex = "\n".join(lines)
    return latex

def process_all_latex_tables(file_paths):
    all_latex = []

    for name, file_path in file_paths.items():
        try:
            df = pd.read_csv(file_path)

            if 'unlearning' not in df.columns:
                print(f"[!] Skipping {name}: no 'unlearning' column.")
                continue

            df = df[df['unlearning'].isin(non_retrain_methods + retrain_methods)]

            numeric_cols = df.select_dtypes(include='number').columns
            grouped = df.groupby('unlearning')[numeric_cols].agg(['mean', 'std'])

            if any(m in grouped.index for m in non_retrain_methods):
                all_latex.append(generate_latex_table(grouped, name, retrain=False))
            if any(m in grouped.index for m in retrain_methods):
                all_latex.append(generate_latex_table(grouped, name, retrain=True))

        except Exception as e:
            print(f"Error processing {name}: {e}")

    return "\n\n".join(all_latex)



files = {    
    # "CIFAR-10 AllCNN": r"C:/Temp/Unlearning/Results/Cifar 10 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR-10 ResNet-18": r"C:/Temp/Unlearning/Results/Cifar 10 Resnet/compiled_results_MIAU.csv",
    # "CIFAR-20 AllCNN": r"C:/Temp/Unlearning/Results/Cifar 20 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR-20 ResNet-18": r"C:/Temp/Unlearning/Results/Cifar 20 Resnet/compiled_results_MIAU.csv",
    # "CIFAR-10 ViT": r"C:/Temp/Unlearning/Results/Cifar 10 ViT/compiled_results_MIAU.csv",
    # "MNIST ResNet-18": r"C:/Temp/Unlearning/Results/MNIST Resnet/compiled_results_MIAU.csv",
    # "MNIST AllCNN": r"C:/Temp/Unlearning/Results/MNIST AllCNN/compiled_results_MIAU.csv",
    # "MUCAC ResNet-18": r"C:/Temp/Unlearning/Results/MUCAC Resnet/compiled_results_MIAU.csv"
    
    "CIFAR10 ResNet Underfitted": r"C:/Temp/Unlearning/Results/Underfitted/compiled_results_MIAU.csv",
    "CIFAR10 ResNet Overfitted": r"C:/Temp/Unlearning/Results/Overfitted/compiled_results_MIAU.csv"
}

latex_output = process_all_latex_tables(files)

with open("compiled_latex_tables_nongeneralized.txt", "w", encoding="utf-8") as f:
    f.write(latex_output)

print("LaTeX tables written to file")
