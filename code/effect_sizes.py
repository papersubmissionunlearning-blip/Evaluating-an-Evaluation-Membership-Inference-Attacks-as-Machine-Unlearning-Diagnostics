import os
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel

RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
OUTPUT_DIR = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"

DATASETS = {
    "CIFAR-10 AllCNN": "Results_AllCNN_Cifar10",
    "CIFAR-10 ResNet-18": "Results_ResNet18_Cifar10",
    "CIFAR-20 AllCNN": "Results_AllCNN_Cifar20",
    "CIFAR-20 ResNet-18": "Results_ResNet18_Cifar20",
    "CIFAR-10 ViT": "Results_ViT_Cifar10",
    "MNIST ResNet-18": "Results_ResNet18_Mnist",
    "MNIST AllCNN": "Results_AllCNN_Mnist",
    "MUCAC ResNet-18": "Results_ResNet18_MUCAC",
    "CIFAR-10 ResNet-18 Saliency": "Results_ResNet18_Cifar10_AdvancedMIA_Xgboost",
}
COMPARISONS = [("retrain50", "retrain25"), ("retrain75", "retrain25"), ("retrain75", "retrain50")]

TEX_HEADER = r"""\begin{table}[!htb]
\caption{Pairwise retraining--level comparisons across datasets. Each row tests whether the MIAU score at the higher retraining level is significantly greater than at the lower level using a one-sided paired $t$-test across 10 seeds. Positive mean differences with $p < 0.05$ support monotonicity; negative mean differences or high $p$-values indicate monotonicity failures.}
\label{tab:retrain-comparisons}
\begin{center}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l l r r r r r r}
\hline
\textbf{Dataset-Model} & \textbf{Comparison} & \textbf{n} & \textbf{Mean diff} & \textbf{95\% CI lower} & \textbf{95\% CI upper} & \textbf{Cohen's d} & \textbf{p-value} \\
\hline \\
"""

TEX_FOOTER = r"""\hline
\end{tabular}%
}
\end{center}
\end{table}
"""


def compute_rows():
    rows = []
    for name, folder in DATASETS.items():
        df = pd.read_csv(os.path.join(RESULTS_ROOT, folder, "compiled_results_MIAU.csv"))
        piv = df.pivot(index="seed", columns="unlearning", values="MIAU")
        for high, low in COMPARISONS:
            pair = piv[[high, low]].dropna()
            diff = pair[high] - pair[low]
            n = len(diff)
            mean, sd = diff.mean(), diff.std(ddof=1)
            half = t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            rows.append({
                "Dataset-Model": name, "Comparison": f"{high} > {low}", "n": n,
                "Mean diff": mean, "95% CI lower": mean - half, "95% CI upper": mean + half,
                "Cohen's d": mean / sd,
                "p-value": ttest_rel(pair[high], pair[low], alternative="greater").pvalue,
            })
    return pd.DataFrame(rows)


def to_latex(df):
    body = []
    for r in df.itertuples(index=False):
        comparison = r[1].replace(" > ", " $>$ ")
        body.append(f"{r[0]} & {comparison} & {r[2]} & {r[3]:.2f} & {r[4]:.2f} & {r[5]:.2f} "
                    f"& {r[6]:.2f} & {r[7]:.3f} \\\\")
    return TEX_HEADER + "\n".join(body) + "\n" + TEX_FOOTER


def main():
    df = compute_rows()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "retrain_level_comparisons.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(OUTPUT_DIR, "retrain_level_comparisons.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(to_latex(df))

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()