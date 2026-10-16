import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

TOFU_CSV = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed\TOFU_Phi1.5_TrueRetrain\compiled_results.csv"
OUTPUT_DIR = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"

# Same style as mia_critique_visualizations.py; drawn at the printed width (5.5in)
plt.rcParams.update({
    'font.family': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'axes.titleweight': 'bold',
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# The TOFU pipeline used alpha = 2 ln(1000), so MUS(0) = 0.1 and MUS(1) = 99.9 exactly
ALPHA = 2 * np.log(1000)

BASE_C, RET_C = '#2c3e50', '#27ae60'
TASKS = [("mia_fvr", "FvR"), ("mia_fvt", "FvT"), ("mia_tvr", "TvR")]
TASK_LABELS = {"FvR": "Forget vs\nRetain", "FvT": "Forget vs\nTest", "TvR": "Test vs\nRetain"}
TASK_COLORS = {"FvR": "#1f77b4", "FvT": "#ff7f0e", "TvR": "#8e44ad"}
LEVELS = ["baseline", "retrain_25", "retrain_50", "retrain_75", "retrain_100"]
LEVEL_X = [0, 25, 50, 75, 100]
LEVEL_LABELS = ["0%\n(Base)", "25%", "50%", "75%", "100%\n(Retrain)"]
APPROX = ["retrain_25", "retrain_50", "retrain_75", "dualoptim"]
APPROX_LABELS = ["Ret. 25%", "Ret. 50%", "Ret. 75%", "DualOptim+"]


def compute_fi(B, R, M):
    # Same as calculate_miau_per_seed.compute_fi
    denom = abs(B - R)
    if denom < 1e-9:
        return 1.0 if abs(M - R) < 1e-9 else -abs(M - R)
    return (abs(B - R) - abs(M - R)) / denom


def compute_musi(f):
    return 100.0 / (1.0 + np.exp(-ALPHA * (np.asarray(f, dtype=float) - 0.5)))


def add_task_scores(df):
    base = df[df["method"] == "baseline"].set_index("seed")
    ret = df[df["method"] == "retrain_100"].set_index("seed")
    out = df.copy()
    for col, t in TASKS:
        B = out["seed"].map(base[col])
        R = out["seed"].map(ret[col])
        out[f"f_{t}"] = [compute_fi(b, r, m) for b, r, m in zip(B, R, out[col])]
        out[f"MUS_{t}"] = compute_musi(out[f"f_{t}"])
    out["MIAU_recomputed"] = out[[f"MUS_{t}" for _, t in TASKS]].mean(axis=1)
    return out


def clipped_err(means, stds):
    means, stds = np.asarray(means), np.asarray(stds)
    return [np.minimum(stds, means), np.minimum(stds, 100 - means)]


def panel_identifiability(ax, df, rng):
    base = df[df["method"] == "baseline"]
    ret = df[df["method"] == "retrain_100"]
    for i, (col, t) in enumerate(TASKS):
        for vals, dx, c in ((base[col], -0.15, BASE_C), (ret[col], 0.15, RET_C)):
            ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(vals)), vals, s=7, color=c, alpha=0.85, lw=0)
        pooled = np.sqrt((base[col].var() + ret[col].var()) / 2)
        snr = abs(ret[col].mean() - base[col].mean()) / pooled
        ax.text(i, 1.02, f"SNR = {snr:.1f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0.5, color="red", ls="--", lw=0.6, alpha=0.6, zorder=0)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([TASK_LABELS[t] for _, t in TASKS])
    ax.set_ylim(0.45, 1.1)
    ax.set_ylabel("MIA accuracy")
    ax.set_title("(a) FM1: Baseline vs Retrain Identifiability")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=BASE_C, label="Baseline"),
                       Line2D([], [], ls="", marker="o", ms=3, color=RET_C, label="Retrain")],
              loc="lower left", frameon=False, handletextpad=0.2)


def panel_noise(ax, df, rng):
    for j, m in enumerate(APPROX):
        v = df.loc[df["method"] == m, "MUS_TvR"]
        ax.scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=TASK_COLORS["TvR"], alpha=0.85, lw=0)
    base = df[df["method"] == "baseline"].set_index("seed")["mia_tvr"]
    ret = df[df["method"] == "retrain_100"].set_index("seed")["mia_tvr"]
    gap = (ret - base).abs().mean()
    ax.text(0.5, 0.5, f"mean |baseline − retrain| on TvR = {gap:.4f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=6,
            bbox=dict(fc="white", ec="0.7", lw=0.4, pad=1.5))
    ax.set_xticks(range(len(APPROX)))
    ax.set_xticklabels(APPROX_LABELS)
    ax.set_ylim(-8, 108)
    ax.set_ylabel("MUS$_{TvR}$ per seed")
    ax.set_title("(b) FM2: Noise Dominates the TvR Score")


def panel_monotonicity(ax, df):
    piv = df.pivot(index="seed", columns="method", values="miau")[LEVELS]
    n_viol = 0
    for _, row in piv.iterrows():
        y = row.to_numpy()
        ax.plot(LEVEL_X, y, color="grey", lw=0.6, alpha=0.7)
        for k in range(1, len(y)):
            if y[k] < y[k - 1]:
                ax.scatter(LEVEL_X[k], y[k], marker="x", color="#e74c3c", s=16, lw=0.9, zorder=3)
                n_viol += 1
    ax.plot(LEVEL_X, piv.mean().to_numpy(), color="black", lw=1.3, marker="o", ms=2.5)
    ax.text(0.03, 0.95, f"{n_viol} violation(s) across {len(piv)} seeds",
            transform=ax.transAxes, va="top", fontsize=6)
    ax.legend(handles=[Line2D([], [], color="grey", lw=0.6, label="Seed"),
                       Line2D([], [], color="black", lw=1.3, marker="o", ms=2.5, label="Mean"),
                       Line2D([], [], ls="", marker="x", color="#e74c3c", label="Violation")],
              loc="lower right", frameon=False, handlelength=1.4)
    ax.set_xticks(LEVEL_X)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Retraining level")
    ax.set_ylabel("MIAU")
    ax.set_title("(c) FM3: Per-Seed Monotonicity")


def panel_rank_consistency(ax, df):
    width = 0.25
    means = {}
    for i, (_, t) in enumerate(TASKS):
        grp = [df.loc[df["method"] == m, f"MUS_{t}"] for m in APPROX]
        means[t] = [g.mean() for g in grp]
        stds = [g.std() for g in grp]
        x = np.arange(len(APPROX)) + (i - 1) * width
        ax.bar(x, means[t], width, yerr=clipped_err(means[t], stds), color=TASK_COLORS[t], label=t,
               error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5))
    lines = []
    for a, b in [("FvR", "FvT"), ("FvR", "TvR"), ("FvT", "TvR")]:
        tau = kendalltau(means[a], means[b])[0]
        lines.append(f"τ({a}, {b}) = " + ("n/a" if np.isnan(tau) else f"{tau:+.2f}"))
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=5.5,
            bbox=dict(fc="white", ec="0.7", lw=0.4, pad=1.5))
    ax.set_xticks(range(len(APPROX)))
    ax.set_xticklabels(APPROX_LABELS)
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Mean MUS$_i$")
    ax.legend(loc="upper right", ncol=3, frameon=False, handlelength=1.0, columnspacing=0.8)
    ax.set_title("(d) FM4: Inter-Task Rank Inconsistency")


def main():
    df = add_task_scores(pd.read_csv(TOFU_CSV))
    drift = (df["MIAU_recomputed"] - df["miau"]).abs().max()
    print(f"max |recomputed MIAU - stored MIAU| = {drift:.2e}")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    panel_identifiability(ax_a, df, rng)
    panel_noise(ax_b, df, rng)
    panel_monotonicity(ax_c, df)
    panel_rank_consistency(ax_d, df)

    fig.suptitle("Failure Modes on TOFU (Phi-1.5, 5 Seeds)", fontsize=7.5, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3, h_pad=1.0, w_pad=1.0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"figure_tofu_failure_modes.{ext}")
        fig.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()