"""
MIAU sensitivity analysis (alpha and task weights) over the 8 main configurations.

Re-computes MIAU from the stored per-seed f_i columns of each
compiled_results_MIAU.csv, for
    alpha   in {5, 10, 13.8, 20, 30}            (weights fixed at 1/3 each)
    weights in {(1/3,1/3,1/3), (.6,.2,.2), (.2,.6,.2), (.2,.2,.6)}   (alpha = 13.8)
and compares the method ranking (by mean MIAU over seeds) with the default
configuration (alpha = 13.8, equal weights) via Kendall's tau.

Input CSVs are only read; nothing is written back to them.

Usage:
    python miau_sensitivity_analysis.py

Outputs (in OUTPUT_DIR):
    miau_sensitivity_detail.csv   one row per experiment x configuration
    miau_sensitivity_summary.csv  one row per configuration (aggregated over the 8 experiments)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"
OUTPUT_DIR = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"
CSV_NAME = "compiled_results_MIAU.csv"

STD_EXPS = {
    "C10 ACNN": "Results_AllCNN_Cifar10",
    "C10 RN": "Results_ResNet18_Cifar10",
    "C10 ViT": "Results_ViT_Cifar10",
    "C20 ACNN": "Results_AllCNN_Cifar20",
    "C20 RN": "Results_ResNet18_Cifar20",
    "MN ACNN": "Results_AllCNN_Mnist",
    "MN RN": "Results_ResNet18_Mnist",
    "MU RN": "Results_ResNet18_MUCAC",
}
# Same ranked set as mia_critique_visualizations.py (baseline/retrain/graded excluded)
RANKED_METHODS = ["amnesiac", "finetune", "teacher", "ssd",
                  "salun", "salun_do", "scrub", "scrub_do", "sfron", "sfron_do"]
TASKS = ["Forget vs Retain", "Forget vs Test", "Test vs Retain"]
F_COLS = [f"f_i ({t} Membership Inference Attack (MIA))" for t in TASKS]

DEFAULT_ALPHA = 13.8
DEFAULT_W = (1 / 3, 1 / 3, 1 / 3)
CONFIGS = (
    [(f"alpha={a}", a, DEFAULT_W) for a in (5, 10, 13.8, 20, 30)]
    + [("w=(.6,.2,.2) FvR", DEFAULT_ALPHA, (0.6, 0.2, 0.2)),
       ("w=(.2,.6,.2) FvT", DEFAULT_ALPHA, (0.2, 0.6, 0.2)),
       ("w=(.2,.2,.6) TvR", DEFAULT_ALPHA, (0.2, 0.2, 0.6))]
)

DETAIL_PATH = os.path.join(OUTPUT_DIR, "miau_sensitivity_detail.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "miau_sensitivity_summary.csv")


def mus(f, alpha):
    # Same as compute_musi in calculate_miau_per_seed.py
    return 100.0 / (1.0 + np.exp(-alpha * (f - 0.5)))


def per_seed_miau(df, alpha, w):
    f = df[F_COLS].to_numpy(dtype=float)
    return pd.Series((mus(f, alpha) * np.asarray(w)).sum(axis=1), index=df.index)


def method_means(df, alpha, w):
    """Mean MIAU over seeds for each ranked method present in df."""
    s = per_seed_miau(df, alpha, w).groupby(df["unlearning"]).mean()
    return s.reindex([m for m in RANKED_METHODS if m in s.index])


def main():
    input_paths = {label: os.path.join(RESULTS_ROOT, folder, CSV_NAME)
                   for label, folder in STD_EXPS.items()}
    protected = {os.path.normcase(os.path.abspath(p)) for p in input_paths.values()}
    for out in (DETAIL_PATH, SUMMARY_PATH):
        if os.path.normcase(os.path.abspath(out)) in protected:
            raise RuntimeError(f"Output path {out} points at an input CSV - refusing to overwrite it.")

    rows = []
    for label, path in input_paths.items():
        if not os.path.exists(path):
            print(f"[!] missing {path}")
            continue
        df = pd.read_csv(path).copy()
        df["unlearning"] = df["unlearning"].replace("ssdtuning", "ssd")
        df = df[df["unlearning"].isin(RANKED_METHODS)].dropna(subset=F_COLS)

        missing = [m for m in RANKED_METHODS if m not in set(df["unlearning"])]
        if missing:
            print(f"[!] {label}: no rows for {missing}")

        if "MIAU" in df.columns:
            drift = np.nanmax(np.abs(per_seed_miau(df, DEFAULT_ALPHA, DEFAULT_W) - df["MIAU"]))
            if drift > 1e-6:
                print(f"[!] {label}: recomputed default MIAU differs from stored MIAU by up to {drift:.4g}")

        ref = method_means(df, DEFAULT_ALPHA, DEFAULT_W)
        for name, a, w in CONFIGS:
            cur = method_means(df, a, w).reindex(ref.index)
            tau, _ = kendalltau(ref.values, cur.values)
            rows.append({
                "experiment": label, "config": name, "n_methods": len(ref),
                "kendall_tau_vs_default": tau,
                "top_method_default": ref.idxmax(), "top_method_config": cur.idxmax(),
                "top_method_unchanged": ref.idxmax() == cur.idxmax(),
                "miau_range_config": cur.max() - cur.min(),
            })

    detail = pd.DataFrame(rows)
    summary = detail.groupby("config", sort=False).agg(
        experiments=("experiment", "count"),
        tau_mean=("kendall_tau_vs_default", "mean"),
        tau_min=("kendall_tau_vs_default", "min"),
        top_unchanged=("top_method_unchanged", "sum"),
        miau_range_mean=("miau_range_config", "mean"),
    ).round(3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH)

    pd.set_option("display.width", 160)
    print(summary)
    print("\nPer-experiment tau (rows: config, cols: experiment):")
    print(detail.pivot(index="config", columns="experiment",
                       values="kendall_tau_vs_default").reindex([c[0] for c in CONFIGS]).round(2))
    print(f"\nSaved: {DETAIL_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()