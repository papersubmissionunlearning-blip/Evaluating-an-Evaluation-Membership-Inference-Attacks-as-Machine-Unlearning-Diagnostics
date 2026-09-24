import pandas as pd
import os

# Map CSV metric names -> LaTeX row labels (use unicode arrow, not LaTeX)
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
retrain_methods = ['retrain25', 'retrain50', 'retrain75', 'retrain']  # keep this exact order

def pretty_method_name(m: str) -> str:
    return m[0].upper() + m[1:] if m else m

def format_mean_std(metric_name, mean, std, digits=2):
    """
    Use literal '±' (unicode), no math mode. 
    Scale non-Accuracy and non-MIAU metrics by 100 (percent).
    """
    if "Accuracy" not in metric_name and metric_name != "MIAU":
        return f"{mean * 100:.{digits}f} ± {std * 100:.{digits}f}"
    else:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

def build_header_row(method_list):
    cells = [r"\multicolumn{1}{c|}{\bf Metric}"]
    for i, m in enumerate(method_list):
        disp = pretty_method_name(m)
        if i < len(method_list) - 1:
            cells.append(fr"\multicolumn{{1}}{{c|}}{{\bf {disp}}}")
        else:
            cells.append(fr"\multicolumn{{1}}{{c}}{{\bf {disp}}}")
    return " & ".join(cells) + r" \\"

def generate_latex_table(df, dataset_name, retrain=False):
    method_list = retrain_methods if retrain else non_retrain_methods
    caption = f"Gradual unlearning on {dataset_name}" if retrain else f"Experiments on {dataset_name}"
    label = f"tab:{dataset_name.lower().replace(' ', '-')}-{'retrain' if retrain else 'noretrain'}"

    rows = []
    hline_inserted = False

    for metric_key, latex_label in metric_map.items():
        if not hline_inserted and "MIA" in metric_key:
            rows.append(r"\hline \\")
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

    col_spec = "l|" + "|".join(["c"] * len(method_list))

    lines = [
        r"\begin{table}[h]",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        r"\begin{center}",
        r"\resizebox{\columnwidth}{!}{%",
        fr"\begin{{tabular}}{{{col_spec}}}",
        build_header_row(method_list),
        r"\hline \\",
    ]
    lines += rows
    lines += [
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table}",
    ]

    return "\n".join(lines)

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

# ----------------- Configure your files here -----------------
files = {
    # "CIFAR-10 AllCNN": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR-10 ResNet-18": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 Resnet/compiled_results_MIAU.csv",
    # "CIFAR-20 AllCNN": r"C:/Temp/Unlearning/Data Appendix/Cifar 20 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR-20 ResNet-18": r"C:/Temp/Unlearning/Data Appendix/Cifar 20 Resnet/compiled_results_MIAU.csv",
    # "CIFAR-10 ViT": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 ViT/compiled_results_MIAU.csv",
    # "MNIST ResNet-18": r"C:/Temp/Unlearning/Data Appendix/MNIST Resnet/compiled_results_MIAU.csv",
    # "MNIST AllCNN": r"C:/Temp/Unlearning/Data Appendix/MNIST AllCNN/compiled_results_MIAU.csv",
    # "MUCAC ResNet-18": r"C:/Temp/Unlearning/Data Appendix/MUCAC Resnet/compiled_results_MIAU.csv"
    
    #  "CIFAR10 ResNet Underfitted": r"C:/Temp/Unlearning/Data Appendix/Underfitted/compiled_results_MIAU.csv",
    # "CIFAR10 ResNet Overfitted": r"C:/Temp/Unlearning/Data Appendix/Overfitted/compiled_results_MIAU.csv"
    
        "CIFAR10 ResNet-18 saliency": r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet Saliency\compiled_results_MIAU.csv",
      "CIFAR20 AllCNN subclass": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN SubClass\compiled_results_MIAU.csv",
    "CIFAR20 AllCNN full class": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN FullClass\compiled_results_MIAU.csv",
}

latex_output = process_all_latex_tables(files)

with open("compiled_latex_tables_iclr_new.txt", "w", encoding="utf-8") as f:
    f.write(latex_output)

print("LaTeX tables written to file")
