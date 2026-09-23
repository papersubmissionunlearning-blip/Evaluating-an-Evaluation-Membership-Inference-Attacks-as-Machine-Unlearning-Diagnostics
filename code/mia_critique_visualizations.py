#!/usr/bin/env python3
"""
Generate publication-quality figures exposing MIA failures.

KEY DESIGN CHOICES:
  FM1 (Figs 1,2,4): Raw MIA scores — testing whether raw scores distinguish baseline from retrain
  FM2 (Fig 7):      Raw MIA scores — measuring signal vs noise in raw scores
  FM3 (Fig 3):      MIAU scores — testing whether the normalized composite increases monotonically
  FM4 (Fig 5):      MUS_i scores — testing whether per-task MIAU components agree on method rankings
  FM5 (Fig 6):      Raw MIA scores — showing that raw MIA behavior changes with training regime
  Summary (Fig 8):  Mixed — consolidating all findings

Requirements: pip install pandas numpy matplotlib scipy
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import ks_2samp
import warnings
import os
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# CSV_FILES = {
#     'cifar10_allcnn':           'cifar10_allcnn_compiled_results_MIAU.csv',
#     'cifar10_resnet':           'cifar10_resnet_compiled_results_MIAU.csv',
#     'cifar10_resnet_saliency':  'cifar10_resnet_saliency_compiled_results_MIAU.csv',
#     'cifar10_vit':              'cifar10_vit_compiled_results_MIAU.csv',
#     'cifar20_allcnn':           'cifar20_allcnn_compiled_results_MIAU.csv',
#     'cifar20_resnet':           'cifar20_resnet_compiled_results_MIAU.csv',
#     'mnist_allcnn':             'mnist_allcnn_compiled_results_MIAU.csv',
#     'mnist_resnet':             'mnist_resnet_compiled_results_MIAU.csv',
#     'mucac_resnet':             'mucac_resnet_compiled_results_MIAU.csv',
#     'overfitted':               'overfitted_model_compiled_results_MIAU.csv',
#     'underfitted':              'underfitted_model_compiled_results_MIAU.csv',
#     'cifar20_allcnn_fullclass': 'cifar20_allcnn_fullclass_compiled_results_MIAU.csv',
#     'cifar20_allcnn_subclass':  'cifar20_allcnn_subclass_compiled_results_MIAU.csv',
# }


CSV_FILES = {
    'cifar10_allcnn':           r"C:\Temp\Unlearning\Data Appendix\Cifar 10 AllCNN\compiled_results_MIAU_0.5.csv", #'cifar10_allcnn_compiled_results_MIAU.csv',
    'cifar10_resnet':           r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet\compiled_results_MIAU_0.5.csv", #'cifar10_resnet_compiled_results_MIAU.csv',
    'cifar10_resnet_saliency':  r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet Saliency\compiled_results_MIAU_0.5.csv", #'cifar10_resnet_saliency_compiled_results_MIAU.csv',
    'cifar10_vit':              r"C:\Temp\Unlearning\Data Appendix\Cifar 10 ViT\compiled_results_MIAU_0.5.csv", #'cifar10_vit_compiled_results_MIAU.csv',
    'cifar20_allcnn':           r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN\compiled_results_MIAU_0.5.csv", #'cifar20_allcnn_compiled_results_MIAU.csv',
    'cifar20_resnet':           r"C:\Temp\Unlearning\Data Appendix\Cifar 20 Resnet\compiled_results_MIAU_0.5.csv", #'cifar20_resnet_compiled_results_MIAU.csv',
    'mnist_allcnn':             r"C:\Temp\Unlearning\Data Appendix\MNIST AllCNN\compiled_results_MIAU_0.5.csv", #'mnist_allcnn_compiled_results_MIAU.csv',
    'mnist_resnet':             r"C:\Temp\Unlearning\Data Appendix\MNIST Resnet\compiled_results_MIAU_0.5.csv", #'mnist_resnet_compiled_results_MIAU.csv',
    'mucac_resnet':             r"C:\Temp\Unlearning\Data Appendix\MUCAC Resnet\compiled_results_MIAU_0.5.csv", #'mucac_resnet_compiled_results_MIAU.csv',
    'overfitted':               r"C:\Temp\Unlearning\Data Appendix\Overfitted\compiled_results_MIAU_0.5.csv", #'overfitted_model_compiled_results_MIAU.csv',
    'underfitted':              r"C:\Temp\Unlearning\Data Appendix\Underfitted\compiled_results_MIAU_0.5.csv", #'underfitted_model_compiled_results_MIAU.csv',
    'cifar20_allcnn_fullclass':  r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN FullClass\compiled_results_MIAU_0.5.csv", #'cifar20_allcnn_fullclass_compiled_results_MIAU.csv',
    'cifar20_allcnn_subclass':   r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN SubClass\compiled_results_MIAU_0.5.csv", #'cifar20_allcnn_subclass_compiled_results_MIAU.csv',
}

OUTPUT_DIR = r'C:\Temp\Unlearning\MIA Critique Paper\figures0.5'

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

COL_MAP = {
    'Forget vs Retain Membership Inference Attack (MIA)': 'MIA_FvR',
    'Forget vs Test Membership Inference Attack (MIA)':   'MIA_FvT',
    'Test vs Retain Membership Inference Attack (MIA)':   'MIA_TvR',
    'Train vs Test Membership Inference Attack (MIA)':    'MIA_TrT',
    'Forget Set Accuracy (Df)':                           'Forget_Acc',
    'Test Accuracy':                                      'Test_Acc',
    'Retain Accuracy':                                    'Retain_Acc',
    'Zero-Retain Forget (ZRF)':                           'ZRF',
    'MUS_i (Forget vs Retain Membership Inference Attack (MIA))': 'MUS_FvR',
    'MUS_i (Forget vs Test Membership Inference Attack (MIA))':   'MUS_FvT',
    'MUS_i (Test vs Retain Membership Inference Attack (MIA))':   'MUS_TvR',
    'f_i (Forget vs Retain Membership Inference Attack (MIA))':   'fi_FvR',
    'f_i (Forget vs Test Membership Inference Attack (MIA))':     'fi_FvT',
    'f_i (Test vs Retain Membership Inference Attack (MIA))':     'fi_TvR',
}

def load_all_data(csv_files):
    dfs = []
    for name, path in csv_files.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping '{name}'")
            continue
        d = pd.read_csv(path)
        d['experiment'] = name
        dfs.append(d)
        print(f"  Loaded {name}: {len(d)} rows")
    if not dfs:
        raise RuntimeError("No CSV files found! Check your paths in CSV_FILES.")
    master = pd.concat(dfs, ignore_index=True).rename(columns=COL_MAP)
    return master

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & STYLE
# ═══════════════════════════════════════════════════════════════

STD_EXPS = ['cifar10_allcnn','cifar10_resnet','cifar10_vit','cifar20_allcnn',
            'cifar20_resnet','mnist_allcnn','mnist_resnet','mucac_resnet']
EXP_LABELS = ['C10 ACNN','C10 RN','C10 ViT','C20 ACNN','C20 RN','MN ACNN','MN RN','MU RN']
EXP_LABELS_NL = ['C10\nACNN','C10\nRN','C10\nViT','C20\nACNN','C20\nRN','MN\nACNN','MN\nRN','MU\nRN']
MIA_COLS   = ['MIA_FvR','MIA_FvT','MIA_TvR']
MUS_COLS   = ['MUS_FvR','MUS_FvT','MUS_TvR']
MIA_NAMES  = ['Forget vs Retain','Forget vs Test','Test vs Retain']
MIA_COLORS = ['#e74c3c','#3498db','#27ae60']
METHOD_COLORS = {'baseline':'#2c3e50','retrain':'#27ae60','amnesiac':'#e74c3c',
                 'finetune':'#3498db','teacher':'#f39c12','ssd':'#9b59b6'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman','DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.8,
})

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_vals(master, exp, method, col):
    return master[(master['experiment']==exp)&(master['unlearning']==method)][col].dropna().values

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), bbox_inches='tight')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: MIA Distributional Overlap (Violin + Box)
# FM1: Raw MIA scores — baseline vs retrain indistinguishable
# ═══════════════════════════════════════════════════════════════
def fig1_mia_distributions(master):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    methods_order = ['baseline','retrain','amnesiac','finetune','teacher','ssd']
    for ax_idx, (col, title) in enumerate(zip(MIA_COLS, MIA_NAMES)):
        ax = axes[ax_idx]
        data = {}
        for m in methods_order:
            vals = master[(master['experiment'].isin(STD_EXPS))&(master['unlearning']==m)][col].dropna().values
            if len(vals) > 0: data[m] = vals
        labels = list(data.keys())
        parts = ax.violinplot([data[m] for m in labels], range(len(labels)),
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, m in zip(parts['bodies'], labels):
            pc.set_facecolor(METHOD_COLORS[m]); pc.set_alpha(0.3); pc.set_edgecolor(METHOD_COLORS[m])
        bp = ax.boxplot([data[m] for m in labels], positions=range(len(labels)),
                       widths=0.15, patch_artist=True, showfliers=False,
                       medianprops=dict(color='black', linewidth=1.5))
        for patch, m in zip(bp['boxes'], labels):
            patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=1, zorder=0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([m.capitalize() for m in labels], rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
        if ax_idx == 0: ax.set_ylabel('MIA Accuracy')
    fig.suptitle('MIA Score Distributions (8 experiments × 10 seeds)', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig1_mia_distributions')
    print("  ✓ Figure 1: MIA Distributional Overlap")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: KS Divergence Heatmap
# FM1: Statistical confirmation — raw MIA scores
# ═══════════════════════════════════════════════════════════════
def fig2_ks_divergence(master):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks_mat = np.zeros((len(STD_EXPS), len(MIA_COLS)))
    pv_mat = np.zeros((len(STD_EXPS), len(MIA_COLS)))
    for i, exp in enumerate(STD_EXPS):
        for j, col in enumerate(MIA_COLS):
            bv = get_vals(master, exp, 'baseline', col)
            rv = get_vals(master, exp, 'retrain', col)
            if len(bv)>0 and len(rv)>0:
                ks, pv = ks_2samp(bv, rv); ks_mat[i,j]=ks; pv_mat[i,j]=pv
    for ax, mat, cmap, title, fmt_func in [
        (ax1, ks_mat, 'RdYlGn', 'KS Statistic', lambda v: f'{v:.2f}'),
        (ax2, pv_mat, 'RdYlGn_r', 'p-value', lambda v: f'{v:.2f}' if v>=0.01 else f'{v:.1e}')]:
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(MIA_NAMES, rotation=20, ha='right')
        ax.set_yticks(range(len(EXP_LABELS_NL))); ax.set_yticklabels(EXP_LABELS_NL, fontsize=8)
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                threshold = mat[ii,jj]>0.5 if 'KS' in title else mat[ii,jj]<0.3
                ax.text(jj, ii, fmt_func(mat[ii,jj]), ha='center', va='center', fontsize=8,
                       color='white' if threshold else 'black')
        ax.set_title(f'{title} (Baseline vs Retrain)', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_fig(fig, 'fig2_ks_divergence')
    print("  ✓ Figure 2: KS Divergence Heatmap")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Monotonicity Failure — USES MIAU SCORES
# FM3: Does the normalized composite increase with more removal?
#
# MIAU is the right metric here because it normalizes against
# baseline and retrain references, giving a 0-100 scale where
# 0 = no forgetting, 100 = ideal forgetting. A decrease in MIAU
# at a higher removal level unambiguously means "further from
# ideal forgetting" — which should never happen.
# ═══════════════════════════════════════════════════════════════
def fig3_monotonicity_failure(master):
    retrain_levels = ['retrain25','retrain50','retrain75','retrain']
    level_labels = ['25%','50%','75%','100%']
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5)); axes = axes.flatten()
    exps_with_violations = 0; total_exps = 0
    for idx, (exp, lbl) in enumerate(zip(STD_EXPS, EXP_LABELS)):
        ax = axes[idx]
        means, stds = [], []
        for rl in retrain_levels:
            v = get_vals(master, exp, rl, 'MIAU')
            means.append(np.mean(v) if len(v)>0 else np.nan)
            stds.append(np.std(v) if len(v)>0 else np.nan)
        means = np.array(means); stds = np.array(stds); ok = ~np.isnan(means)
        exp_has_violation = False
        if ok.sum() > 1:
            total_exps += 1
            x = np.arange(4)
            ax.plot(x[ok], means[ok], 'o-', color='#2c3e50', markersize=5, linewidth=2)
            ax.fill_between(x[ok], (means-stds)[ok], (means+stds)[ok], alpha=0.15, color='#2c3e50')
            vm = means[ok]
            for k in range(len(vm)-1):
                if vm[k+1] < vm[k]:
                    exp_has_violation = True
                    ax.plot(x[ok][k+1], vm[k+1], 'x', color='red', markersize=10,
                            markeredgewidth=2.5, zorder=5)
        if exp_has_violation:
            exps_with_violations += 1
        ax.set_xticks(range(4)); ax.set_xticklabels(level_labels, fontsize=8)
        ax.set_title(lbl, fontsize=9, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.axhline(100, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        if idx % 4 == 0: ax.set_ylabel('MIAU Score')
        if idx >= 4: ax.set_xlabel('Forget-Set Removal Level')
    fig.suptitle(f'MIAU Trajectories Under Graded Retraining '
                 f'(× = monotonicity violation; {exps_with_violations}/{total_exps} experiments fail)',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout()
    save_fig(fig, 'fig3_monotonicity_failure')
    print(f"  ✓ Figure 3: Monotonicity Failure using MIAU ({exps_with_violations}/{total_exps} experiments fail)")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Effect Size (Cohen's d)
# FM1: Raw MIA scores — practical significance
# ═══════════════════════════════════════════════════════════════
def fig4_effect_size(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xp, yv, cl = [], [], []
    for i, exp in enumerate(STD_EXPS):
        for j, (col, clr) in enumerate(zip(MIA_COLS, MIA_COLORS)):
            bv=get_vals(master,exp,'baseline',col); rv=get_vals(master,exp,'retrain',col)
            if len(bv)>1 and len(rv)>1:
                ps=np.sqrt((np.var(bv,ddof=1)+np.var(rv,ddof=1))/2)
                d=abs(np.mean(rv)-np.mean(bv))/ps if ps>0 else 0
                xp.append(i*4+j); yv.append(d); cl.append(clr)
    ax.bar(xp, yv, color=cl, alpha=0.7, width=0.8, edgecolor='white')
    for y, lbl, c in [(0.2,'Small','orange'),(0.5,'Medium','red'),(0.8,'Large','darkred')]:
        ax.axhline(y=y, color=c, linestyle='--', alpha=0.6, linewidth=1)
        ax.text(max(xp), y+0.02, f'{lbl} (d={y})', fontsize=7, color=c, ha='right')
    ax.set_xticks([i*4+1 for i in range(len(STD_EXPS))])
    ax.set_xticklabels(EXP_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Cohen's d"); ax.set_title("Effect Size: Baseline vs Retrain MIA Scores", fontweight='bold')
    ax.legend(handles=[Line2D([0],[0],color=c,lw=6,alpha=0.7,label=n) for c,n in zip(MIA_COLORS,MIA_NAMES)], loc='upper left')
    plt.tight_layout()
    save_fig(fig, 'fig4_effect_size')
    print("  ✓ Figure 4: Effect Size (Cohen's d)")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Rank Contradictions — NOW USES MUS_i SCORES
# FM4: Inter-Task Rank Inconsistency
#
# MUS_i is the per-task MIAU component: for each MIA task, MUS_i
# measures what fraction of the baseline-retrain gap the unlearned
# model has closed, passed through logistic normalization (0-100).
#
# We rank the 4 unlearning methods by their mean MUS_i on each
# task, then check whether these rankings agree across tasks via
# Kendall's τ. This is more principled than using raw MIA scores
# because MUS_i normalizes against baseline and retrain — so a
# higher MUS_i genuinely means "closer to ideal forgetting on
# this task." If the MUS_i rankings disagree across tasks, then
# the three components of MIAU contradict each other about which
# method forgot best.
# ═══════════════════════════════════════════════════════════════
def fig5_rank_contradictions(master):
    methods = ['amnesiac','finetune','teacher','ssd']
    pairs = [('MUS_FvR','MUS_FvT'),('MUS_FvR','MUS_TvR'),('MUS_FvT','MUS_TvR')]
    pair_labels = ['MUS$_{FvR}$ vs MUS$_{FvT}$','MUS$_{FvR}$ vs MUS$_{TvR}$','MUS$_{FvT}$ vs MUS$_{TvR}$']
    tau_mat = np.full((len(STD_EXPS), len(pairs)), np.nan)
    for i, exp in enumerate(STD_EXPS):
        for j, (c1, c2) in enumerate(pairs):
            r1, r2 = [], []
            for m in methods:
                v1 = get_vals(master, exp, m, c1)
                v2 = get_vals(master, exp, m, c2)
                if len(v1)>0 and len(v2)>0:
                    r1.append(np.mean(v1))
                    r2.append(np.mean(v2))
            if len(r1) >= 3:
                tau_mat[i,j], _ = stats.kendalltau(r1, r2)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(tau_mat, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_yticks(range(len(EXP_LABELS))); ax.set_yticklabels(EXP_LABELS)
    for i in range(len(STD_EXPS)):
        for j in range(3):
            v = tau_mat[i,j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                       color='white' if abs(v)>0.5 else 'black', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Kendall's τ")
    ax.set_title("Rank Consistency Between MIAU Components (MUS$_i$)\n"
                 "(low τ = components contradict on which method is best)",
                 fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig5_rank_contradictions')
    print("  ✓ Figure 5: Rank Contradictions using MUS_i (Kendall's τ)")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Regime Sensitivity — RAW MIA INDIVIDUAL SCORES
# FM5: Regime Confounding
#
# Uses raw MIA scores (not MIAU) because the point is to show
# that the raw signal MIAs produce — which is what the community
# actually reports — changes dramatically depending on how well
# the model was trained. The baseline-retrain gap magnitude,
# variance, and even direction shift across regimes, proving
# that MIA behavior is driven by training quality, not just
# unlearning quality.
# ═══════════════════════════════════════════════════════════════
def fig6_regime_sensitivity(master):
    regimes = {'Underfitted':'underfitted','Standard':'cifar10_resnet','Overfitted':'overfitted'}
    regime_colors = {'Underfitted':'#3498db','Standard':'#2c3e50','Overfitted':'#e74c3c'}
    available = master['experiment'].unique()
    for rn, en in list(regimes.items()):
        if en not in available:
            print(f"  ⚠ '{en}' not in data, skipping regime figure"); return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax_idx, (col, title) in enumerate(zip(MIA_COLS, MIA_NAMES)):
        ax = axes[ax_idx]
        for r_idx, (rn, en) in enumerate(regimes.items()):
            bv = get_vals(master, en, 'baseline', col)
            rv = get_vals(master, en, 'retrain', col)
            c = regime_colors[rn]
            if len(bv)>0:
                ax.scatter([r_idx-0.15]*len(bv), bv, c=c, alpha=0.3, s=20)
                ax.scatter(r_idx-0.15, np.mean(bv), c=c, s=80, marker='D',
                          edgecolors='black', linewidths=0.8, zorder=5)
            if len(rv)>0:
                ax.scatter([r_idx+0.15]*len(rv), rv, c=c, alpha=0.3, s=20, marker='^')
                ax.scatter(r_idx+0.15, np.mean(rv), c=c, s=80, marker='^',
                          edgecolors='black', linewidths=0.8, zorder=5)
            if len(bv)>0 and len(rv)>0:
                ax.annotate('', xy=(r_idx+0.15, np.mean(rv)),
                           xytext=(r_idx-0.15, np.mean(bv)),
                           arrowprops=dict(arrowstyle='->', color=c, lw=1.5, alpha=0.7))
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(list(regimes.keys()))
        ax.set_title(title, fontweight='bold')
        if ax_idx == 0: ax.set_ylabel('MIA Accuracy')
    fig.suptitle('MIA Sensitivity to Training Regime (CIFAR-10 ResNet)\n'
                 '(◆=baseline, ▲=retrain)',
                 fontsize=10, fontweight='bold', y=1.04)
    plt.tight_layout()
    save_fig(fig, 'fig6_regime_sensitivity')
    print("  ✓ Figure 6: Regime Sensitivity (Raw MIA Scores)")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Signal-to-Noise Ratio
# FM2: Raw MIA scores — noise exceeds signal
# ═══════════════════════════════════════════════════════════════
def fig7_snr(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(STD_EXPS)); width = 0.25
    for j, (col, lbl, clr) in enumerate(zip(MIA_COLS, ['FvR','FvT','TvR'], MIA_COLORS)):
        snrs = []
        for exp in STD_EXPS:
            bv=get_vals(master,exp,'baseline',col); rv=get_vals(master,exp,'retrain',col)
            if len(bv)>1 and len(rv)>1:
                sig=abs(np.mean(rv)-np.mean(bv)); noise=np.sqrt((np.std(bv)**2+np.std(rv)**2)/2)
                snrs.append(sig/noise if noise>0 else 0)
            else: snrs.append(0)
        ax.bar(x+(j-1)*width, snrs, width, color=clr, alpha=0.7, label=lbl, edgecolor='white')
    ax.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax.text(len(STD_EXPS)-0.5, 1.05, 'SNR=1 (signal=noise)', fontsize=8, ha='right', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(EXP_LABELS_NL, fontsize=9)
    ax.set_ylabel('SNR'); ax.set_title('Signal-to-Noise Ratio: |μ_retrain − μ_baseline| / σ_pooled', fontweight='bold')
    ax.legend(); plt.tight_layout()
    save_fig(fig, 'fig7_snr')
    print("  ✓ Figure 7: Signal-to-Noise Ratio")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: 4-Panel Diagnostic Summary
# ═══════════════════════════════════════════════════════════════
def fig8_diagnostic_summary(master):
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    # (a) Scatter: baseline vs retrain in MIA space
    ax1 = fig.add_subplot(gs[0, 0])
    for exp in STD_EXPS:
        ax1.scatter(get_vals(master,exp,'baseline','MIA_FvR'), get_vals(master,exp,'baseline','MIA_FvT'), c='#2c3e50', alpha=0.2, s=12)
        ax1.scatter(get_vals(master,exp,'retrain','MIA_FvR'), get_vals(master,exp,'retrain','MIA_FvT'), c='#27ae60', alpha=0.2, s=12)
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.3); ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('MIA (FvR)'); ax1.set_ylabel('MIA (FvT)')
    ax1.set_title('(a) Baseline vs Retrain in MIA Space', fontweight='bold', fontsize=10)
    ax1.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2c3e50',label='Baseline',markersize=8),
                        Line2D([0],[0],marker='o',color='w',markerfacecolor='#27ae60',label='Retrain',markersize=8)])
    # (b) MIAU distributions by method
    ax2 = fig.add_subplot(gs[0, 1])
    methods=['amnesiac','finetune','teacher','ssd']; mclrs=['#e74c3c','#3498db','#f39c12','#9b59b6']
    bp = ax2.boxplot([master[(master['experiment'].isin(STD_EXPS))&(master['unlearning']==m)]['MIAU'].dropna().values for m in methods],
                     labels=[m.capitalize() for m in methods], patch_artist=True, showfliers=True, flierprops=dict(markersize=3))
    for patch, c in zip(bp['boxes'], mclrs): patch.set_facecolor(c); patch.set_alpha(0.6)
    ax2.set_ylabel('MIAU Score'); ax2.set_title('(b) MIAU by Method', fontweight='bold', fontsize=10)
    # (c) CDF of retrain MIA scores
    ax3 = fig.add_subplot(gs[1, 0])
    for col, clr, lbl in zip(MIA_COLS, MIA_COLORS, ['FvR','FvT','TvR']):
        rv = master[(master['experiment'].isin(STD_EXPS))&(master['unlearning']=='retrain')][col].dropna().values
        sv = np.sort(rv); cdf = np.arange(1, len(sv)+1)/len(sv)
        ax3.plot(sv, cdf*100, color=clr, linewidth=2, label=lbl)
    ax3.axvline(0.55, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('MIA Accuracy'); ax3.set_ylabel('CDF (%)')
    ax3.set_title('(c) CDF of Retrain MIA Scores', fontweight='bold', fontsize=10); ax3.legend()
    # (d) Std heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    mp = ['amnesiac','finetune','teacher','ssd']
    std_mat = np.zeros((len(STD_EXPS), len(mp)))
    for i, exp in enumerate(STD_EXPS):
        for j, m in enumerate(mp):
            v = get_vals(master, exp, m, 'MIAU'); std_mat[i,j] = np.std(v) if len(v)>1 else 0
    im = ax4.imshow(std_mat, cmap='Reds', aspect='auto')
    ax4.set_xticks(range(4)); ax4.set_xticklabels([m.capitalize() for m in mp])
    ax4.set_yticks(range(len(EXP_LABELS_NL))); ax4.set_yticklabels(EXP_LABELS_NL, fontsize=8)
    for i in range(std_mat.shape[0]):
        for j in range(std_mat.shape[1]):
            ax4.text(j, i, f'{std_mat[i,j]:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if std_mat[i,j]>20 else 'black')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label='Std Dev')
    ax4.set_title('(d) MIAU Std Dev Across Seeds', fontweight='bold', fontsize=10)
    save_fig(fig, 'fig8_summary')
    print("  ✓ Figure 8: 4-Panel Diagnostic Summary")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("  MIAU Paper — Figure Generation Script (v3)")
    print("  FM3: MIAU scores | FM4: MUS_i scores | FM5: Raw MIA")
    print("="*60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("\nLoading CSV files...")
    master = load_all_data(CSV_FILES)
    print(f"  Total: {len(master)} rows from {master['experiment'].nunique()} experiments\n")
    print("Generating figures...")
    #fig1_mia_distributions(master)      # FM1: distributional overlap (raw MIA)
    #fig2_ks_divergence(master)          # FM1: KS test (raw MIA)
    fig3_monotonicity_failure(master)   # FM3: monotonicity (MIAU)
    #fig4_effect_size(master)            # FM1: Cohen's d (raw MIA)
    #fig5_rank_contradictions(master)    # FM4: rank consistency (MUS_i) ← UPDATED
    #fig6_regime_sensitivity(master)     # FM5: regime confounding (raw MIA) ← UPDATED
    #fig7_snr(master)                    # FM2: signal-to-noise (raw MIA)
    #fig8_diagnostic_summary(master)     # Summary (mixed)
    print(f"\n{'='*60}")
    print(f"  All figures saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()