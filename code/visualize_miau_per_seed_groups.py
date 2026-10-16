import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
    "Cifar_10_Resnet": r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet 50\compiled_results_MIAU.csv",
}

# Methods you WANT to consider (order) — we'll drop the excluded ones below.
REQUESTED_METHODS = ["amnesiac", "baseline", "retrain", "retrain25", "retrain50", "retrain75", "finetune", "ssd", "teacher"]
EXCLUDED_METHODS = {"baseline", "retrain", "retrain25", "retrain75", "retrain50"}  # <-- do not visualize these

def pick_metric_column(df: pd.DataFrame, preferred: list[str], user_metric: str | None):
    if user_metric:
        if user_metric in df.columns and pd.api.types.is_numeric_dtype(df[user_metric]):
            return user_metric
        lower_cols = {c.lower(): c for c in df.columns}
        if user_metric.lower() in lower_cols and pd.api.types.is_numeric_dtype(df[lower_cols[user_metric.lower()]]):
            return lower_cols[user_metric.lower()]
    for name in preferred:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    lower_cols = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in lower_cols and pd.api.types.is_numeric_dtype(df[lower_cols[name.lower()]]):
            return lower_cols[name.lower()]
    for c in df.columns:
        if c.lower() != "seed" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def parse_seed_from_filename(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"seed[_-]?(\d+)", str(x))
    return int(m.group(1)) if m else np.nan

def normalize_method_names(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .replace({
                  "fine-tune": "finetune",
                  "fine_tune": "finetune",
                  "teacher-student": "teacher",
                  "amnesiac unlearning": "amnesiac",
                  "baseline_model": "baseline",
              })
    )

def load_all_csvs(files_dict: dict) -> pd.DataFrame:
    dfs, missing = [], []
    for key, path in files_dict.items():
        try:
            if os.path.exists(path):
                d = pd.read_csv(path)
                if "dataset" not in d.columns:
                    d["dataset"] = key.split("_")[0] if "_" in key else key
                if "model" not in d.columns:
                    d["model"] = key.split("_")[1] if "_" in key else "Unknown"
                d["__source_key__"] = key
                dfs.append(d)
            else:
                missing.append((key, path))
        except Exception as e:
            missing.append((key, f"{path} (error: {e})"))
    if not dfs:
        raise FileNotFoundError(
            "No CSVs could be loaded. Check your paths in FILES. "
            f"Missing list: {missing}"
        )
    df = pd.concat(dfs, ignore_index=True)
    return df

def ensure_seed_column(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" in df.columns:
        try:
            df["seed"] = df["seed"].astype(int)
            return df
        except Exception:
            pass
    if "Filename" in df.columns:
        parsed = df["Filename"].apply(parse_seed_from_filename)
        if parsed.notna().any():
            df["seed"] = parsed.astype("Int64")
            return df
    if "seed" not in df.columns:
        df["seed"] = 0
    return df

def summarize_for_plot(df: pd.DataFrame, metric_col: str | None) -> pd.DataFrame:
    cols_needed = {"dataset", "model", "unlearning", "seed"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if metric_col:
        g = (df.groupby(["dataset", "model", "unlearning", "seed"], as_index=False)[metric_col]
                .mean())
        g = g.rename(columns={metric_col: "__value__"})
    else:
        g = (df.groupby(["dataset", "model", "unlearning", "seed"], as_index=False)
                .size()
                .rename(columns={"size": "__value__"}))
    return g

def add_seed_groups_rankwise(tidy: pd.DataFrame, group_size: int = 10) -> pd.DataFrame:
    """
    Within each (dataset, model), sort unique seeds, assign ranks 1..N, then
    compute seed_group = ceil(rank / group_size). Adds columns:
      - __seed_rank__, seed_group (1..), seed_group_label ("Group k (r1–rN)")
    """
    def _rank_and_group(sub):
        # unique sorted seeds
        unique_seeds = pd.Series(sorted(sub["seed"].dropna().unique()))
        rank_map = {int(s): i + 1 for i, s in unique_seeds.items()}
        sub = sub.copy()
        sub["__seed_rank__"] = sub["seed"].map(rank_map).astype(int)

        sub["seed_group"] = ((sub["__seed_rank__"] - 1) // group_size) + 1

        # Build labels like "Group 1 (r1–r10)", "Group 2 (r11–r20)", ...
        def label_from_group(g):
            r_start = (g - 1) * group_size + 1
            r_end = g * group_size
            return f"Group {g} (r{r_start}–r{r_end})"

        sub["seed_group_label"] = sub["seed_group"].apply(label_from_group)
        return sub

    return tidy.groupby(["dataset", "model"], group_keys=False).apply(_rank_and_group)

def plot_group_bars(avg_df: pd.DataFrame, dataset: str, model: str, seed_group_label: str,
                    out_dir: Path, method_order: list[str], ylabel: str):
    """
    avg_df: columns = unlearning, mean_value (already aggregated for THIS group)
    """
    # Keep methods in desired order
    avg_df = avg_df.set_index("unlearning").reindex(method_order).dropna().reset_index()
    if avg_df.empty:
        return None

    # Save the summarized table for this figure
    table_path = out_dir / f"{dataset}_{model}_{seed_group_label.replace(' ', '_').replace('–','-')}_avg.csv"
    avg_df.rename(columns={"mean_value": ylabel}).to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(avg_df))
    ax.bar(x, avg_df["mean_value"].values, width=0.6)
    ax.set_xlabel("Unlearning method")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} • {model} — {ylabel} — {seed_group_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df["unlearning"], rotation=15, ha="right")
    fig.tight_layout()

    img_name = f"{dataset}_{model}_{seed_group_label.replace(' ', '_').replace('–','-')}_bars.png"
    img_path = out_dir / img_name
    fig.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return img_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=r"C:\Temp\Unlearning\Charts",
                        help="Output folder for charts and summaries (default: ./unlearning_barcharts)")
    parser.add_argument("--metric", type=str, default="MIAU",
                        help="Name of the metric column to plot (if omitted, auto-detect; if none found, plot counts)")
    parser.add_argument("--group_size", type=int, default=10, help="Seeds per group (default: 10)")
    args = parser.parse_args()

    out_dir = Path(args.out or "./unlearning_barcharts")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_csvs(FILES)

    if "unlearning" not in df.columns:
        raise ValueError("CSV must contain an 'unlearning' column.")
    df["unlearning"] = normalize_method_names(df["unlearning"])
    df = ensure_seed_column(df)

    # Determine method order; then drop excluded methods
    present = sorted(df["unlearning"].dropna().unique().tolist())
    method_order_full = [m for m in REQUESTED_METHODS if m in present] + [m for m in present if m not in REQUESTED_METHODS]
    method_order = [m for m in method_order_full if m not in EXCLUDED_METHODS]

    # Filter dataframe to only the methods we want to visualize
    df = df[df["unlearning"].isin(method_order)].copy()
    if df.empty:
        raise ValueError("No rows to plot after filtering out excluded methods. Check your CSV and method names.")

    preferred_metric_names = ["MIAU", "MUS", "ZRF", "AUC", "ROC_AUC", "Accuracy", "ACC", "Score", "score", "metric", "value"]
    metric_col = pick_metric_column(df, preferred_metric_names, args.metric)
    ylabel = metric_col if metric_col else "Count"

    # Per-seed summary
    tidy = summarize_for_plot(df, metric_col)

    # Add seed-grouping by rank within (dataset, model)
    tidy = add_seed_groups_rankwise(tidy, group_size=args.group_size)

    # For each (dataset, model, seed_group), average across the seeds in that group
    # Result: one row per (dataset, model, seed_group, unlearning)
    grouped_avg = (tidy
        .groupby(["dataset", "model", "seed_group", "seed_group_label", "unlearning"], as_index=False)["__value__"]
        .mean()
        .rename(columns={"__value__": "mean_value"}))

    saved = []
    for (dataset, model, seed_group, seed_group_label), sub in grouped_avg.groupby(["dataset", "model", "seed_group", "seed_group_label"]):
        dataset_s = str(dataset).replace(" ", "_")
        model_s = str(model).replace(" ", "_")
        img = plot_group_bars(
            avg_df=sub[["unlearning", "mean_value"]],
            dataset=dataset_s,
            model=model_s,
            seed_group_label=str(seed_group_label),
            out_dir=out_dir,
            method_order=method_order,
            ylabel=ylabel,
        )
        if img:
            saved.append(str(img))

    # If you truly have 50 seeds per (dataset, model), this will yield exactly 5 visuals.
    print("Metric used:", ylabel)
    print("Charts saved:")
    for p in saved:
        print(" -", p)
    print("Folder:", out_dir.resolve())

if __name__ == "__main__":
    main()
