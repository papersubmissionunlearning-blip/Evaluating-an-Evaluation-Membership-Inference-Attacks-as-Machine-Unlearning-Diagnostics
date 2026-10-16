import pandas as pd
import os

metric_map = {
    "Retain Accuracy": "Retain Accuracy",
    "Test Accuracy": "Test Accuracy",
    "Forget Set Accuracy (Df)": "Forget Accuracy",
    "Zero-Retain Forget (ZRF)": "ZRF Score",
    "Forget vs Retain Membership Inference Attack (MIA)": "MIA (Forget vs Retain)",
    "Forget vs Test Membership Inference Attack (MIA)": "MIA (Forget vs Test)",
    "Test vs Retain Membership Inference Attack (MIA)": "MIA (Test vs Retain)",
    "Train vs Test Membership Inference Attack (MIA)": "MIA (Train vs Test)",
    "MIAU": "MIAU",
}

# Display names for methods
method_display_names = {
    'baseline': 'Baseline',
    'amnesiac': 'Amnesiac',
    'finetune': 'Finetune',
    'teacher': 'Teacher',
    'ssd': 'SSD',
    'retrain25': 'Retrain 25\\%',
    'retrain50': 'Retrain 50\\%',
    'retrain75': 'Retrain 75\\%',
    'retrain': 'Retrain',
    'salun': 'SalUn',
    'scrub': 'SCRUB',
    'sfron': 'SFRon',
    'salun_do': 'SalUn+DualOptim',
    'scrub_do': 'SCRUB+DualOptim',
    'sfron_do': 'SFRon+DualOptim',
}

non_retrain_methods = ['baseline', 'amnesiac', 'finetune', 'teacher', 'ssd']
retrain_methods = ['retrain25', 'retrain50', 'retrain75', 'retrain']
advanced_methods = ['salun', 'salun_do', 'scrub', 'scrub_do', 'sfron', 'sfron_do']


def format_mean_std(metric_name, mean, std, digits=4):
    if "Accuracy" not in metric_name and metric_name != "MIAU":
        return f"{mean * 100:.{digits}f} ± {std * 100:.{digits}f}"
    else:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"


def metric_group(metric_key):
    """Rows are split by a \\midrule into: other metrics, MIA metrics, MIAU."""
    if metric_key == "MIAU":
        return 2
    if "MIA" in metric_key:
        return 1
    return 0


def assemble_table(caption, label, column_headers, rows):
    """
    Booktabs table without vertical rules, as the NeurIPS template asks.

    rows: list of (metric_key, cells) where cells[0] is the row label.
    The table is set in \\small and only shrunk when still wider than the text.
    """
    col_spec = "l" + "c" * len(column_headers)
    header_parts = [r"\multicolumn{1}{c}{\bf Metric}"]
    header_parts += [rf"\multicolumn{{1}}{{c}}{{\bf {h}}}" for h in column_headers]

    lines = [
        r"\begin{table}[!htb]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
        r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
        r"\midrule",
    ]

    previous_group = None
    for metric_key, cells in rows:
        group = metric_group(metric_key)
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        previous_group = group
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table(df, dataset_name, label_prefix, method_list, table_type="retrain"):
    """
    Generate a LaTeX table for one experiment.
    table_type: 'retrain', 'noretrain', 'advanced'
    """
    if table_type == "retrain":
        caption = f"Gradual unlearning on {dataset_name}"
        label_suffix = "retrain"
    elif table_type == "noretrain":
        caption = f"Experiments on {dataset_name}"
        label_suffix = "noretrain"
    elif table_type == "advanced":
        caption = f"SalUn, SCRUB, SFRon with and without DualOptim on {dataset_name}"
        label_suffix = "advanced"
    else:
        caption = f"Results on {dataset_name}"
        label_suffix = "results"

    label = f"tab:{label_prefix}-{label_suffix}"
    headers = [method_display_names.get(m, m.capitalize()) for m in method_list]

    rows = []
    for metric_key, latex_label in metric_map.items():
        # Skip ZRF for advanced method tables
        if "ZRF" in metric_key and table_type == "advanced":
            continue

        cells = [rf"\textbf{{{latex_label}}}" if metric_key == "MIAU" else latex_label]
        for method in method_list:
            try:
                mean = df.loc[method][metric_key]['mean']
                std = df.loc[method][metric_key]['std']
                cells.append(format_mean_std(metric_key, mean, std, digits=4))
            except KeyError:
                cells.append("–")
        rows.append((metric_key, cells))

    return assemble_table(caption, label, headers, rows)


def parse_tofu_summary(file_path):
    """
    Parse the TOFU summary table text file format.
    Returns a dict of {metric: {method: "mean±std"}} and list of methods.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = {}
    methods = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('TOFU'):
            continue
        
        parts = line.split()
        
        # Header line with method names
        if parts[0] == 'Metric':
            methods = parts[1:]
            continue
        
        # Data lines - handle multi-word metric names
        # Check MIAU first since it contains 'MIA'
        if 'MIAU' in line:
            metric_name = 'MIAU'
            values = parts[1:]
        elif 'MIA' in line:
            # MIA (FvR), MIA (FvT), MIA (TvR)
            metric_match = line.split(')')
            metric_name = metric_match[0] + ')'
            values = metric_match[1].split()
        else:
            # Forget Loss, Retain Loss, Test Loss
            metric_name = parts[0] + ' ' + parts[1]
            values = parts[2:]
        
        data[metric_name] = {}
        for i, method in enumerate(methods):
            if i < len(values):
                data[metric_name][method] = values[i]
    
    return data, methods


def generate_tofu_latex_table(file_path, dataset_name="TOFU Phi-1.5"):
    """
    Generate LaTeX table from TOFU summary file.
    """
    data, methods = parse_tofu_summary(file_path)
    
    # Map TOFU method names to standard format
    method_name_map = {
        'Baseline': 'Baseline',
        'DualOptim+': 'DualOptim+',
        'Ret.25%': r'Retrain 25\%',
        'Ret.50%': r'Retrain 50\%',
        'Ret.75%': r'Retrain 75\%',
        'Ret.100%': 'Retrain',
    }
    
    caption = f"Experiments on {dataset_name}"
    label = f"tab:{dataset_name.lower().replace(' ', '-').replace('.', '')}-results"

    # Use mapped name or escape % for LaTeX
    headers = [method_name_map.get(m, m.replace('%', r'\%')) for m in methods]

    # Metric display order and names
    metric_order = [
        ("Forget Loss", "Forget Loss"),
        ("Retain Loss", "Retain Loss"),
        ("Test Loss", "Test Loss"),
        ("MIA (FvR)", "MIA (Forget vs Retain)"),
        ("MIA (FvT)", "MIA (Forget vs Test)"),
        ("MIA (TvR)", "MIA (Test vs Retain)"),
        ("MIAU", "MIAU"),
    ]
    
    rows = []
    for metric_key, latex_label in metric_order:
        row = [rf"\textbf{{{latex_label}}}" if metric_key == "MIAU" else latex_label]

        for method in methods:
            if metric_key in data and method in data[metric_key]:
                # Parse and reformat to 4 decimal places
                raw_value = data[metric_key][method]
                try:
                    if '±' in raw_value:
                        mean_str, std_str = raw_value.split('±')
                        mean_val = float(mean_str)
                        std_val = float(std_str)
                        cell = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        cell = f"{float(raw_value):.4f}"
                except ValueError:
                    cell = raw_value.replace('±', ' ± ')
                row.append(cell)
            else:
                row.append("–")

        rows.append((metric_key, row))

    return assemble_table(caption, label, headers, rows)


def process_all_latex_tables(file_paths):
    """
    Process all files and generate LaTeX tables.
    """
    all_latex = []

    for name, (label_prefix, file_path) in file_paths.items():
        try:
            df = pd.read_csv(file_path)

            if 'unlearning' not in df.columns:
                print(f"[!] Skipping {name}: no 'unlearning' column.")
                continue

            numeric_cols = df.select_dtypes(include='number').columns
            grouped = df.groupby('unlearning')[numeric_cols].agg(['mean', 'std'])

            # Generate standard tables
            available_non_retrain = [m for m in non_retrain_methods if m in grouped.index]
            available_retrain = [m for m in retrain_methods if m in grouped.index]
            
            if available_non_retrain:
                all_latex.append(generate_latex_table(grouped, name, label_prefix, available_non_retrain, table_type="noretrain"))
            if available_retrain:
                all_latex.append(generate_latex_table(grouped, name, label_prefix, available_retrain, table_type="retrain"))
            
            # Advanced tables are emitted for whichever experiments have the rows
            available_advanced = [m for m in advanced_methods if m in grouped.index]
            if available_advanced:
                all_latex.append(generate_latex_table(grouped, name, label_prefix, available_advanced, table_type="advanced"))

        except Exception as e:
            print(f"Error processing {name}: {e}")

    return "\n\n".join(all_latex)


RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"

# caption name -> (label prefix, csv path)
# Captions follow the NeurIPS rule (lower case except proper nouns). Label
# prefixes are the ones the paper already \ref's, so they don't follow the
# caption spelling for every experiment.
files = {
    "CIFAR-10 AllCNN": ("cifar-10-allcnn", os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar10", "compiled_results_MIAU.csv")),
    "CIFAR-10 ResNet-18": ("cifar-10-resnet-18", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10", "compiled_results_MIAU.csv")),
    "CIFAR-10 overfitted ResNet-18": ("cifar10-resnet-overfitted", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_Overfitted", "compiled_results_MIAU.csv")),
    "CIFAR-10 underfitted ResNet-18": ("cifar10-resnet-underfitted", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_Underfitted", "compiled_results_MIAU.csv")),
    "CIFAR-10 ResNet-18 shadow": ("cifar10-resnet-18-shadow", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Shadow", "compiled_results_MIAU.csv")),
    "CIFAR-10 ResNet-18 quantile": ("cifar10-resnet-18-quantile", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Quantile", "compiled_results_MIAU.csv")),
    "CIFAR-10 ResNet-18 LiRA": ("cifar10-resnet-18-lira", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Lira", "compiled_results_MIAU.csv")),
    "CIFAR-10 ResNet-18 saliency": ("cifar10-resnet-18-saliency", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Xgboost", "compiled_results_MIAU.csv")),
    "CIFAR-10 ViT": ("cifar-10-vit", os.path.join(RESULTS_ROOT, "Results_ViT_Cifar10", "compiled_results_MIAU.csv")),
    "CIFAR-20 AllCNN": ("cifar-20-allcnn", os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar20", "compiled_results_MIAU.csv")),
    "CIFAR-20 AllCNN full class": ("cifar20-allcnn-full-class", os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar20_FullClass", "compiled_results_MIAU.csv")),
    "CIFAR-20 AllCNN subclass": ("cifar20-allcnn-subclass", os.path.join(RESULTS_ROOT, "Results_AllCNN_Cifar20_SubClass", "compiled_results_MIAU.csv")),
    "CIFAR-20 ResNet-18": ("cifar-20-resnet-18", os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar20", "compiled_results_MIAU.csv")),
    "MNIST AllCNN": ("mnist-allcnn", os.path.join(RESULTS_ROOT, "Results_AllCNN_Mnist", "compiled_results_MIAU.csv")),
    "MNIST ResNet-18": ("mnist-resnet-18", os.path.join(RESULTS_ROOT, "Results_ResNet18_Mnist", "compiled_results_MIAU.csv")),
    "MUCAC ResNet-18": ("mucac-resnet-18", os.path.join(RESULTS_ROOT, "Results_ResNet18_MUCAC", "compiled_results_MIAU.csv")),
}

# TOFU summary file
tofu_file = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\TOFU_Phi1.5_TrueRetrain\summary_table.txt"

if __name__ == "__main__":
    latex_output = process_all_latex_tables(files)

    # Add TOFU table
    try:
        tofu_latex = generate_tofu_latex_table(tofu_file, "TOFU Phi-1.5")
        latex_output += "\n\n" + tofu_latex
        print("Added TOFU Phi-1.5 table")
    except Exception as e:
        print(f"Error processing TOFU file: {e}")

    output_path = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures\compiled_latex_tables_neurips.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_output)

    print(f"LaTeX tables written to {output_path}")
