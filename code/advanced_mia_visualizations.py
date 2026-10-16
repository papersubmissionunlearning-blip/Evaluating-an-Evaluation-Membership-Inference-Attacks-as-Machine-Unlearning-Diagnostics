#!/usr/bin/env python3
"""
Figures for the stronger-attack experiments (Saliency, Shadow, Quantile, LiRA) on CIFAR-10 ResNet-18.

  Dot plot:     raw MIA scores per method, one marker per attack, every seed drawn
  Monotonicity: the Figure 3 MIAU-vs-removal-level chart, one panel per attack
  Monotonicity (ablation): the same chart with MIAU computed without Test vs Retain

Shadow, Quantile and LiRA have 3 seeds, so the dot plot shows each seed rather
than a distribution. Saliency is shown with all 10 seeds.

Requirements: pip install pandas numpy matplotlib scipy
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Importing also applies the shared rcParams (fonts, sizes, TrueType embedding)
import mia_critique_visualizations as mcv
from mia_critique_visualizations import (
    CSV_FILES, MIA_COLS, MIA_NAMES, METHOD_LABELS,
    load_all_data, get_vals, save_fig, fig3_monotonicity_failure,
)

RESULTS_ROOT = r"C:\Temp\Unlearning\Data Appendix\Results_Fixed\Results_Fixed"

ATTACK_CSV_FILES = {
    'saliency': CSV_FILES['cifar10_resnet_saliency'],
    'shadow':   os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Shadow", "compiled_results_MIAU.csv"),
    'quantile': os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Quantile", "compiled_results_MIAU.csv"),
    'lira':     os.path.join(RESULTS_ROOT, "Results_ResNet18_Cifar10_AdvancedMIA_Lira", "compiled_results_MIAU.csv"),
}
ABLATION_ATTACK_CSV_FILES = {
    name: path.replace('compiled_results_MIAU.csv', 'compiled_results_ablation_MIAU.csv')
    for name, path in ATTACK_CSV_FILES.items()
}
ATTACKS        = ['saliency', 'shadow', 'quantile', 'lira']
ATTACK_LABELS  = ['Saliency', 'Shadow', 'Quantile', 'LiRA']
ATTACK_COLORS  = ['#2c3e50', '#e67e22', '#8e44ad', '#16a085']
ATTACK_MARKERS = ['o', 's', '^', 'D']
SEEDS = [1, 2, 3]

# Methods run under every attack (the advanced runs have no SalUn/SCRUB/SFRon)
DOT_METHODS = ['baseline', 'retrain', 'amnesiac', 'finetune', 'teacher', 'ssd']


# ═══════════════════════════════════════════════════════════════
# FIGURE A: Attack comparison dot plot (raw MIA scores)
# ═══════════════════════════════════════════════════════════════
def fig_attack_dotplot(master, name='fig_advanced_mia_dotplot', figsize=(5.5, 2.3)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x = np.arange(len(DOT_METHODS))
    offsets = np.linspace(-0.3, 0.3, len(ATTACKS))
    for ax, col, title in zip(axes, MIA_COLS, MIA_NAMES):
        for k, (atk, clr, mk) in enumerate(zip(ATTACKS, ATTACK_COLORS, ATTACK_MARKERS)):
            for i, m in enumerate(DOT_METHODS):
                v = get_vals(master, atk, m, col)
                if len(v) == 0:
                    continue
                xi = x[i] + offsets[k]
                ax.scatter(np.full(len(v), xi), v, s=6, marker=mk, color=clr, alpha=0.5, linewidths=0)
                ax.hlines(np.mean(v), xi - 0.08, xi + 0.08, color=clr, linewidth=1.1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
        for b in x[:-1]:
            ax.axvline(b + 0.5, color='lightgray', linewidth=0.4)
        ax.set_xlim(-0.5, len(DOT_METHODS) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in DOT_METHODS], rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title(title, fontweight='bold')
    axes[0].set_ylabel('MIA Score')
    handles = [Line2D([0], [0], marker=mk, color=clr, linestyle='none', markersize=3.5, label=lbl)
               for lbl, clr, mk in zip(ATTACK_LABELS, ATTACK_COLORS, ATTACK_MARKERS)]
    handles.append(Line2D([0], [0], color='black', linewidth=1.1, label='Mean'))
    plt.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.84))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.84), ncol=len(handles),
               fontsize=6, handlelength=1.2, handletextpad=0.5, columnspacing=1.5, frameon=False)
    fig.suptitle('MIA Scores Under Stronger Attacks (CIFAR-10 ResNet-18)',
                 fontsize=7.5, fontweight='bold', y=0.99, va='top')
    save_fig(fig, name)
    print(f"  ✓ Figure ({name}): Attack comparison dot plot")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  MIAU Paper — Advanced MIA Figures")
    print("=" * 60)
    os.makedirs(mcv.OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(mcv.OUTPUT_DIR)}")
    print("\nLoading CSV files...")
    master = load_all_data(ATTACK_CSV_FILES)
    master = master[(master['experiment'] == 'saliency') | master['seed'].isin(SEEDS)]
    print(f"  Saliency: all seeds; other attacks: seeds {SEEDS}: {len(master)} rows\n")
    print("Generating figures...")
    fig_attack_dotplot(master)
    fig3_monotonicity_failure(master, name='fig_advanced_mia_monotonicity', figsize=(4.7, 1.45),
                              exps=ATTACKS, labels=ATTACK_LABELS, ncols=4)

    print("\nLoading ablation CSV files (MIAU without Test vs Retain)...")
    ablation = load_all_data(ABLATION_ATTACK_CSV_FILES)
    ablation = ablation[(ablation['experiment'] == 'saliency') | ablation['seed'].isin(SEEDS)]
    fig3_monotonicity_failure(ablation, name='fig_advanced_mia_monotonicity_0.5_alpha', figsize=(4.7, 1.45),
                              exps=ATTACKS, labels=ATTACK_LABELS, ncols=4)
    print(f"\n{'=' * 60}")
    print(f"  All figures saved to: {os.path.abspath(mcv.OUTPUT_DIR)}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
