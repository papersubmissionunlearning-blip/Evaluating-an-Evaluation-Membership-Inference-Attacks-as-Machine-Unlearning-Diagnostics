"""
TOFU MIAU Experiment with True Retraining

This experiment uses the TOFU dataset with TRUE retraining from base Phi-1.5.
Since the base model has never seen TOFU data, fine-tuning IS equivalent to
fresh learning - giving us proper retrain references.

Key features:
1. Uses TOFU data (established benchmark)
2. Trains from base Phi-1.5 (not pre-fine-tuned on TOFU)
3. True retraining (model only knows what it's trained on)
4. Faithful DualOptim+ port (AdamWDecouplePlus optimizer + ME/GD alternation)
5. Balanced membership-inference evaluation with cross-validation
6. Checkpoint loading/resumption for long experiments
7. GPU-optimized for Colab A100

Author: MIA Critique Paper
Adapted from DualOptim+ (https://github.com/CityU-MLO/DualOptimPlus)
"""

import os
import json
import math
import random
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm


@dataclass
class TOFURetrainConfig:
    """Configuration for TOFU true retraining MIAU experiment"""

    # Model (base Phi-1.5, NOT fine-tuned on TOFU)
    model_path: str = "microsoft/phi-1_5"
    tokenizer_path: str = "microsoft/phi-1_5"

    # Data paths
    data_path: str = "../DualOptimPlus/data/tofu"
    split: str = "forget05"
    retain_source: str = "retain50"

    # Sample sizes (2400 total). Using forget05 (2000 rows) with 5 seeds gives
    # completely disjoint forget sets (400 × 5 = 2000). The complementary retain50
    # has 2000 rows; after splitting into 1000-row halves, we use all 1000 each.
    max_forget_samples: int = 400   # paper-exact; with 5 seeds, fills 2000-row pool exactly
    max_retain_samples: int = 1000  # all of one 1000-row half of retain50
    max_test_samples: int = 1000    # all of the other half
    disjoint_forget: bool = True    # partition forget pool into non-overlapping chunks per seed

    # Retraining config (baseline + graded models)
    seeds: List[int] = None  # Will be set to [1-10]
    num_epochs: int = 3      # Epochs for baseline/graded training
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # Prompt format (official DualOptim+ config/model_config.yaml, phi1.5 entry)
    question_start_tag: str = "<|user|>\n"
    question_end_tag: str = "<|end|>\n<|assistant|>\n"
    answer_tag: str = ""
    max_length: int = 500  # official forget.py

    # --- DualOptim+ hyperparameters ---
    # Source: scripts/tofu_phi1-5/me_gd_dual.sh + config/phi1-5_tofu.yaml
    # NOTE: the shipped shell script overrides several yaml defaults. Where they
    # disagree (forget_coeff, mask, base betas) the script wins, since it is the
    # runnable artifact that produced the published numbers.
    forget_lr: float = 1e-5
    retain_lr: float = 1e-5
    forget_coeff: float = 1.0          # script: forget_coeff=1.0 (yaml default 0.1)
    regularization_coeff: float = 1.0
    forget_freq: int = 1               # 1 forget optimizer step ...
    retain_freq: int = 5               # ... per 5 retain optimizer steps
    gradient_accumulation_steps: int = 10  # 4 x 10 = 40, matches 4 x 5 x 2 GPUs
    unlearn_max_steps: int = 300       # script: max_steps=300 (overrides epochs)
    unlearn_epochs: int = 5            # only used if unlearn_max_steps is None
    adam_beta1: float = 0.9            # delta first moment
    adam_beta2: float = 0.95           # delta second moment
    base_beta1: float = 0.9            # shared base first moment (script, not yaml)
    base_beta2: float = 0.95           # shared base second moment (script, not yaml)
    adam_epsilon: float = 1e-8
    dual_alpha: float = 1.0            # accepted by the optimizer, unused in fp32 path
    optimizer_state_dtype: str = "float32"  # "bfloat16" halves optimizer memory

    # Precision
    use_bf16: bool = True

    # Graded retraining levels (fraction of forget data REMOVED)
    graded_levels: List[float] = None  # [0.25, 0.5, 0.75, 1.0]

    # MIAU config
    alpha: float = 13.8136
    mia_cv_folds: int = 5

    # Paths
    results_dir: str = "TOFU Phi-1.5 TrueRetrain"
    checkpoint_dir: str = "checkpoints_tofu_v2"
    # Each Phi-1.5 state dict is ~5.7 GB and a full run produces 60 of them.
    # Turning this off matches the official script's save_checkpoint=false, at
    # the cost of having to retrain from the first seed after a disconnect.
    save_checkpoints: bool = True

    # Device (GPU for Colab A100)
    device: str = "cuda"

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [1, 2, 3, 4, 5]  # 5 seeds for disjoint 400-sample forget sets
        if self.graded_levels is None:
            self.graded_levels = [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# DualOptim+ optimizer
# ---------------------------------------------------------------------------

def _adam_decouple_plus_step(g, p, state1, beta1, state2, beta2,
                             state1_base, beta1_base, state2_base, beta2_base,
                             eps, step, base_step, lr, weight_decay):
    """
    Single-parameter update, transcribed from the fp32 branch of
    ``adam_decouple_plus_step`` in DualOptim+ ``trainer/custom_optimizer_plus.py``
    (the "v5" variant that is live in the released code).

    The shared base state tracks the raw gradient across both objectives; each
    mode keeps a delta state that absorbs only the residual left over after the
    base estimate is subtracted. The two are recombined at update time.
    """
    if weight_decay != 0.0:
        p.mul_(1 - lr * weight_decay)

    # Base states are bias-corrected against base_step - 1 because they are
    # rolled forward at the *end* of the previous step, not the start of this one.
    if base_step > 1:
        s1_base_hat = state1_base / (1 - beta1_base ** (base_step - 1))
        s2_base_hat = state2_base / (1 - beta2_base ** (base_step - 1))
    else:
        s1_base_hat = state1_base
        s2_base_hat = state2_base

    state1.mul_(beta1).add_(g - s1_base_hat, alpha=1 - beta1)
    state2.mul_(beta2).add_(torch.square(g) - s2_base_hat, alpha=1 - beta2)

    # Delta states use their own per-mode step counter.
    s1_hat = state1 / (1 - beta1 ** step)
    s2_hat = state2 / (1 - beta2 ** step)

    exp_avg = s1_hat + s1_base_hat
    # .abs() before .sqrt(): the recombined second moment can go negative
    # because the delta term is a signed residual.
    denom = (s2_hat + s2_base_hat).abs().sqrt().add_(eps)

    p.addcdiv_(exp_avg, denom, value=-lr)

    state1_base.mul_(beta1_base).add_(g, alpha=1 - beta1_base)
    state2_base.mul_(beta2_base).addcmul_(g, g, value=1 - beta2_base)


class AdamWDecouplePlus(torch.optim.Optimizer):
    """
    Faithful reimplementation of DualOptim+'s ``AdamWDecouplePlus`` (32-bit path).

    The upstream class subclasses a bitsandbytes optimizer, but with
    ``optim_bits=32`` and ``percentile_clipping=100`` every bitsandbytes code
    path is either bypassed or a no-op whose result is discarded, so a plain
    ``torch.optim.Optimizer`` reproduces it exactly without the dependency.

    Which objective a step belongs to is decided *inside* the optimizer from its
    own step counter, not from the loss it is handed. With
    ``switch_freq_1=1, switch_freq_2=5`` the pattern is F R R R R R repeating, so
    the training loop must feed batches in exactly that order.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95, 0.9, 0.95), alpha=1.0,
                 eps=1e-8, weight_decay=1e-2, lr_ratio_1=1.0, lr_ratio_2=1.0,
                 switch_freq_1=1, switch_freq_2=1, state_dtype=torch.float32):
        if len(betas) != 4:
            raise ValueError(f"betas must be a 4-tuple (beta1, beta2, base_beta1, base_beta2), got {betas}")
        defaults = dict(lr=lr, betas=tuple(betas), alpha=alpha, eps=eps,
                        weight_decay=weight_decay, lr_ratio_1=lr_ratio_1,
                        lr_ratio_2=lr_ratio_2, switch_freq_1=switch_freq_1,
                        switch_freq_2=switch_freq_2)
        super().__init__(params, defaults)
        self.state_dtype = state_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta1_base, beta2_base = group["betas"]
            sf1, sf2 = group["switch_freq_1"], group["switch_freq_2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWDecouplePlus does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    z = lambda: torch.zeros_like(p, dtype=self.state_dtype, memory_format=torch.preserve_format)
                    state["step"] = 0
                    state["step1"] = 0
                    state["step2"] = 0
                    state["state1_base"] = z()
                    state["state2_base"] = z()
                    state["state1_1"] = z()
                    state["state2_1"] = z()
                    state["state1_2"] = z()
                    state["state2_2"] = z()

                # Mode is read BEFORE the counter is incremented, so step 0 is a
                # forget step and the cycle is F R R R R R.
                mode = 1 if (state["step"] % (sf1 + sf2)) < sf1 else 2
                ratio = group["lr_ratio_1"] if mode == 1 else group["lr_ratio_2"]

                state["step"] += 1
                base_step = state["step"]
                if mode == 1:
                    state["step1"] += 1
                    step = state["step1"]
                else:
                    state["step2"] += 1
                    step = state["step2"]

                _adam_decouple_plus_step(
                    g=grad.to(self.state_dtype),
                    p=p,
                    state1=state["state1_1"] if mode == 1 else state["state1_2"],
                    beta1=beta1,
                    state2=state["state2_1"] if mode == 1 else state["state2_2"],
                    beta2=beta2,
                    state1_base=state["state1_base"],
                    beta1_base=beta1_base,
                    state2_base=state["state2_base"],
                    beta2_base=beta2_base,
                    eps=group["eps"],
                    step=step,
                    base_step=base_step,
                    lr=ratio * group["lr"],
                    weight_decay=group["weight_decay"],
                )

        return loss

    def current_mode(self) -> int:
        """Mode (1=forget, 2=retain) the next step() call will apply."""
        for group in self.param_groups:
            sf1, sf2 = group["switch_freq_1"], group["switch_freq_2"]
            for p in group["params"]:
                state = self.state[p]
                if state:
                    return 1 if (state["step"] % (sf1 + sf2)) < sf1 else 2
        return 1


def build_param_groups(model, weight_decay: float):
    """Split parameters the way HuggingFace's Trainer does: no decay on biases
    and normalization weights (all of which are 1-D)."""
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSON or JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    data.append(json.loads(line))
    return data


def normalize_qa(item: Dict) -> Dict:
    """Reduce the various TOFU record shapes to a plain question/answer pair."""
    if 'question' in item and 'answer' in item:
        return {"question": item['question'], "answer": item['answer']}
    if 'input' in item and 'output' in item:
        return {"question": item['input'], "answer": item['output']}
    raise KeyError(f"Cannot find a question/answer pair in record with keys {sorted(item)}")


def load_tofu_data_for_seed(config: TOFURetrainConfig, seed: int,
                           seed_index: int = 0, num_seeds: int = 1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build the forget / retain / test split for one seed.

    If config.disjoint_forget is True, the forget pool is partitioned into
    num_seeds equal chunks (shuffled with a fixed seed for reproducibility),
    and this seed gets chunk seed_index. This ensures zero overlap between seeds.

    The retain pool is partitioned into two disjoint halves before sampling, so
    the test set is drawn from records that are guaranteed never to appear in
    training.
    """
    rng = random.Random(seed)

    forget_path = os.path.join(config.data_path, f"{config.split}.json")
    forget_all = [normalize_qa(x) for x in load_jsonl(forget_path)]

    retain_path = os.path.join(config.data_path, f"{config.retain_source}.json")
    retain_all = [normalize_qa(x) for x in load_jsonl(retain_path)]

    # Validate pool sizes
    total_forget_needed = config.max_forget_samples * num_seeds if config.disjoint_forget else config.max_forget_samples
    if len(forget_all) < total_forget_needed:
        raise ValueError(
            f"{forget_path} has {len(forget_all)} rows but disjoint_forget={config.disjoint_forget} "
            f"with {num_seeds} seeds × {config.max_forget_samples} samples needs {total_forget_needed}."
        )
    if not config.disjoint_forget and len(forget_all) == config.max_forget_samples:
        logging.getLogger(__name__).warning(
            f"max_forget_samples={config.max_forget_samples} equals the pool size, so "
            f"every seed trains on the identical forget set. Seed variance comes only "
            f"from the retain/test partition and the graded removal order."
        )
    if len(retain_all) < 2 * max(config.max_retain_samples, config.max_test_samples):
        raise ValueError(
            f"{retain_path} has {len(retain_all)} rows, too few to draw "
            f"{config.max_retain_samples} retain and {config.max_test_samples} test "
            "from two disjoint halves."
        )

    # Forget set: disjoint partition or random sample
    if config.disjoint_forget:
        # Shuffle with fixed seed so partition is deterministic across runs
        shuffled = list(forget_all)
        random.Random(42).shuffle(shuffled)
        chunk_size = config.max_forget_samples
        start = seed_index * chunk_size
        forget_data = shuffled[start:start + chunk_size]
        logging.getLogger(__name__).info(
            f"Disjoint forget: seed {seed} gets chunk {seed_index} (rows {start}-{start + chunk_size - 1})"
        )
    else:
        forget_data = rng.sample(forget_all, config.max_forget_samples)

    # Retain/test: partition into two halves, then sample
    indices = list(range(len(retain_all)))
    rng.shuffle(indices)
    midpoint = len(indices) // 2
    retain_pool = [retain_all[i] for i in indices[:midpoint]]
    test_pool = [retain_all[i] for i in indices[midpoint:]]

    retain_data = rng.sample(retain_pool, config.max_retain_samples)
    test_data = rng.sample(test_pool, config.max_test_samples)

    return forget_data, retain_data, test_data


def assert_supervised_tokens(splits: Dict[str, List[Dict]], tokenizer, config: TOFURetrainConfig):
    """
    A sample whose question fills the whole window has every label masked, which
    makes its cross-entropy 0/0 = NaN and silently poisons every mean it feeds.
    Cheaper to catch once up front than to discover it in the summary table.
    """
    for name, rows in splits.items():
        dataset = TOFUDataset(rows, tokenizer, config)
        empty = sum(1 for i in range(len(dataset)) if int((dataset[i]['labels'] != -100).sum()) == 0)
        if empty:
            raise ValueError(
                f"{empty}/{len(rows)} records in the {name} split have no unmasked "
                f"label tokens at max_length={config.max_length}; their loss would be NaN. "
                "Raise max_length."
            )


def nested_forget_keep_order(forget_data: List[Dict], seed: int) -> List[int]:
    """
    One shuffled ordering of the forget set per seed. Each graded level keeps a
    prefix of it, so the records kept at 50% removal are a subset of those kept
    at 25% removal and the only thing varying between levels is how much more
    was dropped.
    """
    order = list(range(len(forget_data)))
    random.Random(seed + 99991).shuffle(order)
    return order


class TOFUDataset(Dataset):
    """
    Tokenizes Q&A pairs exactly as DualOptim+'s
    ``convert_raw_forget_data_to_model_format`` does with ``mask=True``:

    - prompt is ``<|user|>\\n{q}<|end|>\\n<|assistant|>\\n{a}``
    - the sequence is padded to ``max_length`` with EOS
    - labels mask the question tokens and all padding except the first EOS,
      so the model still learns where the answer stops
    """

    def __init__(self, data: List[Dict], tokenizer, config: TOFURetrainConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = normalize_qa(self.data[idx])
        cfg = self.config

        prompt = cfg.question_start_tag + item['question'] + cfg.question_end_tag
        full_text = prompt + cfg.answer_tag + item['answer']

        # Counted on the untruncated prompt, so it can exceed max_length for a
        # very long question. Left unclamped, the slice assignment below would
        # grow the label list past max_length and break batch collation.
        num_question_tokens = min(
            len(self.tokenizer.tokenize(prompt, add_special_tokens=True)),
            self.max_length,
        )

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = list(encoded['input_ids'])
        attention_mask = list(encoded['attention_mask'])
        pad_length = self.max_length - len(input_ids)
        eos = self.tokenizer.eos_token_id

        if pad_length > 0:
            labels = input_ids + [eos] + [-100] * (pad_length - 1)
            input_ids = input_ids + [eos] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        else:
            labels = list(input_ids)

        labels[:num_question_tokens] = [-100] * num_question_tokens

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: TOFURetrainConfig) -> logging.Logger:
    """Setup logging"""
    os.makedirs(config.results_dir, exist_ok=True)

    log_file = os.path.join(config.results_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def autocast_ctx(config: TOFURetrainConfig):
    if config.use_bf16 and config.device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def compute_loss_per_sample(model, data: List[Dict], tokenizer, config: TOFURetrainConfig) -> List[float]:
    """
    Per-sample cross-entropy over answer tokens only.

    Scoring against ``labels=input_ids`` instead would make this mostly a
    measure of how confidently the model predicts EOS after EOS, since padding
    fills most of a 500-token sequence.
    """
    model.eval()
    losses = []

    dataset = TOFUDataset(data, tokenizer, config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            with autocast_ctx(config):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(float(outputs.loss.item()))

    return losses


def _balance(member_losses: np.ndarray, non_member_losses: np.ndarray, seed: int):
    """Subsample the larger side so chance accuracy is exactly 0.5."""
    n = min(len(member_losses), len(non_member_losses))
    rng = np.random.RandomState(seed)
    m_idx = rng.choice(len(member_losses), n, replace=False)
    n_idx = rng.choice(len(non_member_losses), n, replace=False)
    X = np.concatenate([member_losses[m_idx], non_member_losses[n_idx]]).reshape(-1, 1)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


def compute_mia_metrics(member_losses: np.ndarray, non_member_losses: np.ndarray,
                        seed: int, n_folds: int = 5) -> Tuple[float, float]:
    """
    Balanced membership-inference attack on per-sample loss.

    Returns (balanced_accuracy, auc), both averaged over stratified folds and
    both scored on held-out data. Fitting and scoring on the same rows, on
    unbalanced classes, lets the classifier score the majority-class ratio
    without learning anything.
    """
    member_losses = np.asarray(member_losses, dtype=float)
    non_member_losses = np.asarray(non_member_losses, dtype=float)

    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return 0.5, 0.5

    X, y = _balance(member_losses, non_member_losses, seed)

    n_per_class = len(y) // 2
    folds = min(n_folds, n_per_class)
    if folds < 2:
        return 0.5, 0.5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(balanced_accuracy_score(y[test_idx], pred))
        prob = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))

    return float(np.mean(accs)), float(np.mean(aucs))


def evaluate_model(model, forget_data, retain_data, test_data, tokenizer,
                   config: TOFURetrainConfig, seed: int) -> Dict[str, float]:
    """Loss + balanced MIA metrics for one model."""
    forget_losses = compute_loss_per_sample(model, forget_data, tokenizer, config)
    retain_losses = compute_loss_per_sample(model, retain_data, tokenizer, config)
    test_losses = compute_loss_per_sample(model, test_data, tokenizer, config)

    f = np.array(forget_losses)
    r = np.array(retain_losses)
    t = np.array(test_losses)

    mia_fvr, auc_fvr = compute_mia_metrics(f, r, seed, config.mia_cv_folds)
    mia_fvt, auc_fvt = compute_mia_metrics(f, t, seed, config.mia_cv_folds)
    mia_tvr, auc_tvr = compute_mia_metrics(t, r, seed, config.mia_cv_folds)

    return {
        "forget_loss_mean": float(np.mean(forget_losses)),
        "retain_loss_mean": float(np.mean(retain_losses)),
        "test_loss_mean": float(np.mean(test_losses)),
        "mia_fvr": mia_fvr,
        "mia_fvt": mia_fvt,
        "mia_tvr": mia_tvr,
        "auc_fvr": auc_fvr,
        "auc_fvt": auc_fvt,
        "auc_tvr": auc_tvr,
    }


def compute_fi(baseline_mia: float, retrain_mia: float, unlearn_mia: float) -> float:
    """
    Compute gap closure fraction f_i.

    f_i = (|B - R| - |M - R|) / |B - R|

    Edge case handling (from Code Appendix):
    - If |B - R| ~ 0: return 1.0 if ideal, else negative
    """
    gap = abs(baseline_mia - retrain_mia)

    if gap < 1e-9:
        if abs(unlearn_mia - retrain_mia) < 1e-9:
            return 1.0
        else:
            return -abs(unlearn_mia - retrain_mia)

    return (gap - abs(unlearn_mia - retrain_mia)) / gap


def compute_miau_score(baseline_mia: float, retrain_mia: float, unlearn_mia: float, alpha: float) -> Tuple[float, float]:
    """Compute MIAU score with logistic normalization"""
    fi = compute_fi(baseline_mia, retrain_mia, unlearn_mia)
    mus_i = 100.0 / (1.0 + np.exp(-alpha * (fi - 0.5)))
    return fi, mus_i


def aggregate_miau(baseline: Dict[str, float], retrain: Dict[str, float],
                   target: Dict[str, float], alpha: float) -> float:
    """Equal-weighted mean of the three per-task MIAU scores."""
    scores = []
    for key in ("mia_fvr", "mia_fvt", "mia_tvr"):
        _, mus = compute_miau_score(baseline[key], retrain[key], target[key], alpha)
        scores.append(mus)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_data: List[Dict], tokenizer, config: TOFURetrainConfig,
                num_epochs: int, desc: str = "Training") -> torch.nn.Module:
    """Fine-tune a model on the given data"""
    model.train()

    dataset = TOFUDataset(train_data, tokenizer, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = num_epochs * len(loader)
    pbar = tqdm(total=total_steps, desc=desc)

    for epoch in range(num_epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            with autocast_ctx(config):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    model.eval()
    return model


def create_fresh_model(config: TOFURetrainConfig) -> torch.nn.Module:
    """Load a fresh base model"""
    config_hf = AutoConfig.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        config=config_hf,
        torch_dtype=torch.float32,
    ).to(config.device)
    return model


def load_or_create_model(checkpoint_path: str, config: TOFURetrainConfig,
                         logger: logging.Logger) -> Tuple[torch.nn.Module, bool]:
    """
    Load model from checkpoint if it exists, otherwise create fresh model.

    Returns:
        Tuple of (model, loaded_from_checkpoint)
    """
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading existing checkpoint: {checkpoint_path}")
        model = create_fresh_model(config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        model.eval()
        return model, True
    else:
        logger.info(f"No checkpoint found, creating fresh model")
        return create_fresh_model(config), False


def compute_me_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ME (Maximum Entropy) loss, transcribed from ``get_me_loss`` in DualOptim+
    ``trainer/losses.py``.

    KL divergence between the model's next-token distribution and the uniform
    distribution, averaged over unmasked positions. The trainer *minimizes*
    ``forget_coeff * me_loss``; there is no sign flip anywhere upstream.
    """
    num_labels = logits.shape[-1]  # vocab_size

    shifted_labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels)

    loss_mask = (shifted_labels != -100).view(-1).float()

    kl_div = F.kl_div(
        (soft_outputs + 1e-12).log(),
        uniform_dist,
        reduction="none",
    ).sum(-1)

    masked_kl_div = kl_div * loss_mask
    return masked_kl_div.sum() / (loss_mask.sum() + 1e-12)


def _infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def run_dualoptim_unlearning(model, forget_data: List[Dict], retain_data: List[Dict],
                             tokenizer, config: TOFURetrainConfig, seed: int,
                             logger: logging.Logger) -> torch.nn.Module:
    """
    ME+GD unlearning driven by the real DualOptim+ optimizer.

    Settings follow ``scripts/tofu_phi1-5/me_gd_dual.sh``: optim_cfg=dual_adam_plus,
    alternate=true, forget_coeff=1.0, regularization_coeff=1.0, retain_freq=5,
    lr=forget_lr=1e-5, betas=(0.9, 0.95, 0.9, 0.95), max_steps=300.

    The optimizer decides forget-vs-retain from its own internal counter, so the
    batch order here (1 forget optimizer step, then 5 retain ones) has to match
    ``switch_freq_1``/``switch_freq_2`` exactly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.train()

    forget_loader = DataLoader(TOFUDataset(forget_data, tokenizer, config),
                               batch_size=config.batch_size, shuffle=True, drop_last=False)
    retain_loader = DataLoader(TOFUDataset(retain_data, tokenizer, config),
                               batch_size=config.batch_size, shuffle=True, drop_last=False)
    forget_batches = _infinite_batches(forget_loader)
    retain_batches = _infinite_batches(retain_loader)

    ga = config.gradient_accumulation_steps
    cycle = config.forget_freq + config.retain_freq
    effective_batch = config.batch_size * ga

    # Mirrors forget.py: one epoch of warmup, where an "epoch" spans the whole
    # forget/retain cycle rather than just the forget batches.
    steps_per_epoch = int(cycle * math.ceil(len(forget_data) / effective_batch))
    if config.unlearn_max_steps is not None:
        max_steps = config.unlearn_max_steps
    else:
        max_steps = config.unlearn_epochs * steps_per_epoch
    warmup_steps = max(1, steps_per_epoch)

    state_dtype = torch.bfloat16 if config.optimizer_state_dtype == "bfloat16" else torch.float32

    optimizer = AdamWDecouplePlus(
        build_param_groups(model, config.weight_decay),
        lr=config.retain_lr,
        betas=(config.adam_beta1, config.adam_beta2, config.base_beta1, config.base_beta2),
        alpha=config.dual_alpha,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        lr_ratio_1=config.forget_lr / config.retain_lr,
        lr_ratio_2=1.0,
        switch_freq_1=config.forget_freq,
        switch_freq_2=config.retain_freq,
        state_dtype=state_dtype,
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        remaining = max_steps - current_step
        return max(0.0, remaining / max(1, max_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    forget_epochs = (max_steps / cycle) * effective_batch / max(1, len(forget_data))
    logger.info(
        f"DualOptim+ seed {seed}: {max_steps} optimizer steps "
        f"({max_steps // cycle} forget / {max_steps - max_steps // cycle} retain), "
        f"effective batch {effective_batch}, warmup {warmup_steps}, "
        f"~{forget_epochs:.1f} epochs over the forget set"
    )

    pbar = tqdm(total=max_steps, desc=f"DualOptim+ (seed {seed})")

    for opt_step in range(max_steps):
        is_forget = (opt_step % cycle) < config.forget_freq
        source = forget_batches if is_forget else retain_batches

        running = 0.0
        for _ in range(ga):
            batch = next(source)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            with autocast_ctx(config):
                if is_forget:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = config.forget_coeff * compute_me_loss(outputs.logits, labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = config.regularization_coeff * outputs.loss

            (loss / ga).backward()
            running += float(loss.item()) / ga

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.update(1)
        pbar.set_postfix({"phase": "forget" if is_forget else "retain", "loss": f"{running:.4f}"})

    pbar.close()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "method", "seed",
    "forget_loss_mean", "retain_loss_mean", "test_loss_mean",
    "mia_fvr", "mia_fvt", "mia_tvr",
    "auc_fvr", "auc_fvt", "auc_tvr",
    "miau",
]

SUMMARY_METRICS = [
    ("Forget Loss", "forget_loss_mean"),
    ("Retain Loss", "retain_loss_mean"),
    ("Test Loss", "test_loss_mean"),
    ("MIA (FvR)", "mia_fvr"),
    ("MIA (FvT)", "mia_fvt"),
    ("MIA (TvR)", "mia_tvr"),
    ("MIAU", "miau"),
]


def save_per_seed_txt(results_dir: str, method: str, seed: int, metrics: dict):
    """Save per-seed results to TXT file"""
    filename = f"{method}_tofu_phi1.5_seed_{seed}.txt"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"Method: {method}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Dataset: TOFU (True Retrain)\n")
        f.write(f"Model: Phi-1.5 (base)\n")
        f.write("=" * 40 + "\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")


def save_summary_table(results_dir: str, summary_stats: dict, methods: List[str]):
    """Save summary table"""
    filepath = os.path.join(results_dir, "summary_table.txt")

    with open(filepath, 'w') as f:
        f.write("TOFU Phi-1.5 True Retrain MIAU Results Summary\n")
        f.write("=" * 120 + "\n\n")

        header = f"{'Metric':<15}"
        for method in methods:
            header += f"{method:<22}"
        f.write(header + "\n")
        f.write("-" * (15 + 22 * len(methods)) + "\n")

        for metric in summary_stats:
            row = f"{metric:<15}"
            for method in methods:
                if method in summary_stats[metric]:
                    mean = summary_stats[metric][method]["mean"]
                    std = summary_stats[metric][method]["std"]
                    row += f"{mean:.5f}±{std:.5f}".ljust(22)
                else:
                    row += "N/A".ljust(22)
            f.write(row + "\n")

        f.write("\n" + "=" * 120 + "\n")


def save_compiled_csv(results_dir: str, rows: List[dict]):
    """Save compiled results CSV"""
    filepath = os.path.join(results_dir, "compiled_results.csv")

    if not rows:
        return

    with open(filepath, 'w') as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in CSV_COLUMNS) + "\n")


def log_memory_estimate(model, config: TOFURetrainConfig, logger: logging.Logger):
    """DualOptim+ keeps six optimizer state tensors per parameter, so warn early
    if that will not fit rather than OOM an hour into a seed."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = 2 if config.optimizer_state_dtype == "bfloat16" else 4
    gib = (n_params * (4 + 4 + 6 * state_bytes)) / 1e9

    logger.info(f"Trainable parameters: {n_params / 1e9:.2f}B")
    logger.info(f"Estimated DualOptim+ peak (weights + grads + 6 states): {gib:.1f} GB")

    if config.device == "cuda" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gib > 0.9 * total:
            logger.warning(
                f"Estimated {gib:.1f} GB exceeds 90% of the {total:.1f} GB on this GPU. "
                "Set optimizer_state_dtype='bfloat16' to roughly halve optimizer memory."
            )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(config: TOFURetrainConfig):
    """Run the TOFU true retraining MIAU experiment"""
    logger = setup_logging(config)

    logger.info("=" * 70)
    logger.info("TOFU TRUE RETRAINING MIAU EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_path} (base, NOT pre-trained on TOFU)")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(f"Samples: {config.max_forget_samples} forget, {config.max_retain_samples} retain, {config.max_test_samples} test")
    logger.info(f"Graded levels: {config.graded_levels}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} accum")
    logger.info(f"Max length: {config.max_length}")
    logger.info(f"DualOptim+: forget_coeff={config.forget_coeff}, retain_freq={config.retain_freq}, "
                f"betas=({config.adam_beta1}, {config.adam_beta2}, {config.base_beta1}, {config.base_beta2}), "
                f"max_steps={config.unlearn_max_steps}")
    logger.info("NOTE: Using base Phi-1.5 - TRUE retraining (model never saw TOFU)")
    logger.info("=" * 70)

    # Check GPU
    if config.device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            config.device = "cpu"

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Method names
    graded_method_names = [f"Ret.{int(level*100)}%" for level in config.graded_levels]
    all_methods = ["Baseline", "DualOptim+"] + graded_method_names

    all_results = {
        "config": asdict(config),
        "seeds": {}
    }

    all_csv_rows = []
    memory_logged = False

    num_seeds = len(config.seeds)
    for seed_index, seed in enumerate(config.seeds):
        seed_key = str(seed)

        logger.info(f"\n{'='*70}")
        logger.info(f"SEED {seed} ({seed_index + 1}/{num_seeds})")
        logger.info(f"{'='*70}")

        # Load TOFU data for this seed. Drawn once and reused by every method so
        # all models in a seed see identical forget/retain/test records.
        logger.info(f"Loading TOFU data (seed {seed})...")
        forget_data, retain_data, test_data = load_tofu_data_for_seed(
            config, seed, seed_index=seed_index, num_seeds=num_seeds
        )
        forget_order = nested_forget_keep_order(forget_data, seed)
        assert_supervised_tokens(
            {"forget": forget_data, "retain": retain_data, "test": test_data}, tokenizer, config
        )

        logger.info(f"  Forget: {len(forget_data)}, Retain: {len(retain_data)}, Test: {len(test_data)}")

        all_results["seeds"][seed_key] = {
            "baseline": {},
            "dualoptim": {},
            "graded": {}
        }

        # ===== PHASE 1: Baseline (train on ALL data) =====
        logger.info("\n--- Phase 1: Baseline Training ---")
        baseline_ckpt_path = os.path.join(config.checkpoint_dir, f"baseline_seed{seed}.pt")
        baseline_model, baseline_loaded = load_or_create_model(baseline_ckpt_path, config, logger)

        if not memory_logged:
            log_memory_estimate(baseline_model, config, logger)
            memory_logged = True

        if not baseline_loaded:
            all_train_data = forget_data + retain_data
            baseline_model = train_model(baseline_model, all_train_data, tokenizer, config,
                                         config.num_epochs, f"Baseline (seed {seed})")
            if config.save_checkpoints:
                torch.save(baseline_model.state_dict(), baseline_ckpt_path)
                logger.info(f"Saved baseline checkpoint: {baseline_ckpt_path}")
        else:
            logger.info("Skipping baseline training (checkpoint exists)")

        baseline_metrics = evaluate_model(baseline_model, forget_data, retain_data, test_data,
                                          tokenizer, config, seed)

        logger.info(f"Baseline MIA - FvR: {baseline_metrics['mia_fvr']:.4f}, "
                    f"FvT: {baseline_metrics['mia_fvt']:.4f}, TvR: {baseline_metrics['mia_tvr']:.4f}")

        all_results["seeds"][seed_key]["baseline"] = baseline_metrics

        # ===== PHASE 2: Graded Retraining =====
        logger.info("\n--- Phase 2: Graded Retraining ---")

        graded_results = {}
        for level in config.graded_levels:
            level_key = f"retrain_{int(level*100)}"
            logger.info(f"\nRetrain {int(level*100)}%...")

            graded_ckpt_path = os.path.join(config.checkpoint_dir, f"{level_key}_seed{seed}.pt")
            graded_model, graded_loaded = load_or_create_model(graded_ckpt_path, config, logger)

            if not graded_loaded:
                # True retraining: keep a nested prefix of the shuffled forget order
                n_forget_keep = int(len(forget_data) * (1 - level))
                forget_subset = [forget_data[i] for i in forget_order[:n_forget_keep]]
                train_subset = retain_data + forget_subset

                graded_model = train_model(graded_model, train_subset, tokenizer, config,
                                           config.num_epochs, f"Ret.{int(level*100)}% (seed {seed})")
                if config.save_checkpoints:
                    torch.save(graded_model.state_dict(), graded_ckpt_path)
                    logger.info(f"Saved graded checkpoint: {graded_ckpt_path}")
            else:
                logger.info(f"Skipping {level_key} training (checkpoint exists)")

            graded_results[level_key] = evaluate_model(graded_model, forget_data, retain_data,
                                                       test_data, tokenizer, config, seed)

            logger.info(f"  MIA - FvR: {graded_results[level_key]['mia_fvr']:.4f}, "
                        f"FvT: {graded_results[level_key]['mia_fvt']:.4f}, "
                        f"TvR: {graded_results[level_key]['mia_tvr']:.4f}")

            del graded_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_results["seeds"][seed_key]["graded"] = graded_results

        # ===== PHASE 3: DualOptim+ Unlearning =====
        logger.info("\n--- Phase 3: DualOptim+ Unlearning ---")

        dualoptim_ckpt_path = os.path.join(config.checkpoint_dir, f"dualoptim_seed{seed}.pt")

        if os.path.exists(dualoptim_ckpt_path):
            logger.info(f"Loading existing DualOptim+ checkpoint: {dualoptim_ckpt_path}")
            unlearned_model = create_fresh_model(config)
            unlearned_model.load_state_dict(torch.load(dualoptim_ckpt_path, map_location=config.device))
            unlearned_model.eval()
            logger.info("Skipping DualOptim+ training (checkpoint exists)")
        else:
            unlearn_model = create_fresh_model(config)
            unlearn_model.load_state_dict(baseline_model.state_dict())

            unlearned_model = run_dualoptim_unlearning(
                unlearn_model, forget_data, retain_data, tokenizer, config, seed, logger
            )
            if config.save_checkpoints:
                torch.save(unlearned_model.state_dict(), dualoptim_ckpt_path)
                logger.info(f"Saved DualOptim+ checkpoint: {dualoptim_ckpt_path}")

        dualoptim_metrics = evaluate_model(unlearned_model, forget_data, retain_data, test_data,
                                           tokenizer, config, seed)

        logger.info(f"DualOptim+ MIA - FvR: {dualoptim_metrics['mia_fvr']:.4f}, "
                    f"FvT: {dualoptim_metrics['mia_fvt']:.4f}, TvR: {dualoptim_metrics['mia_tvr']:.4f}")

        all_results["seeds"][seed_key]["dualoptim"] = dualoptim_metrics

        # ===== PHASE 4: Compute MIAU =====
        logger.info("\n--- Phase 4: MIAU Computation ---")

        retrain_ref = graded_results["retrain_100"]

        # The baseline is the reference B, so it closes none of the gap and
        # f_i = 0 by definition, giving 100 / (1 + exp(alpha/2)) ~ 0.10. This is
        # evaluated directly rather than through compute_fi, whose |B - R| < 1e-9
        # branch would report f_i = 1 (perfect unlearning) for a model that did
        # no unlearning at all whenever a task happens to give B == R.
        baseline_metrics["miau"] = float(100.0 / (1.0 + np.exp(-config.alpha * (0.0 - 0.5))))

        dualoptim_metrics["miau"] = aggregate_miau(baseline_metrics, retrain_ref,
                                                   dualoptim_metrics, config.alpha)
        logger.info(f"DualOptim+ MIAU: {dualoptim_metrics['miau']:.2f}")

        for level in config.graded_levels:
            level_key = f"retrain_{int(level*100)}"
            graded_results[level_key]["miau"] = aggregate_miau(
                baseline_metrics, retrain_ref, graded_results[level_key], config.alpha
            )
            logger.info(f"  {level_key} MIAU: {graded_results[level_key]['miau']:.2f}")

        # Write per-seed files now that MIAU is known
        save_per_seed_txt(config.results_dir, "baseline", seed, baseline_metrics)
        all_csv_rows.append({"method": "baseline", "seed": seed, **baseline_metrics})

        for level in config.graded_levels:
            level_key = f"retrain_{int(level*100)}"
            save_per_seed_txt(config.results_dir, level_key, seed, graded_results[level_key])
            all_csv_rows.append({"method": level_key, "seed": seed, **graded_results[level_key]})

        save_per_seed_txt(config.results_dir, "dualoptim", seed, dualoptim_metrics)
        all_csv_rows.append({"method": "dualoptim", "seed": seed, **dualoptim_metrics})

        # Save partial results
        partial_path = os.path.join(config.results_dir, "partial_results.json")
        with open(partial_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        save_compiled_csv(config.results_dir, all_csv_rows)

        # Cleanup
        del baseline_model, unlearned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== PHASE 5: Aggregate and Save =====
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    save_compiled_csv(config.results_dir, all_csv_rows)

    # Build summary statistics
    summary_stats = {}

    for metric, key in SUMMARY_METRICS:
        summary_stats[metric] = {}

        # Baseline
        if key != "miau":
            vals = [all_results["seeds"][str(s)]["baseline"][key] for s in config.seeds]
            summary_stats[metric]["Baseline"] = {"mean": np.mean(vals), "std": np.std(vals)}
        else:
            summary_stats[metric]["Baseline"] = {"mean": 0.10, "std": 0.00}

        # DualOptim+
        if key in all_results["seeds"][str(config.seeds[0])]["dualoptim"]:
            vals = [all_results["seeds"][str(s)]["dualoptim"][key] for s in config.seeds]
            summary_stats[metric]["DualOptim+"] = {"mean": np.mean(vals), "std": np.std(vals)}

        # Graded levels
        for level in config.graded_levels:
            level_key = f"retrain_{int(level*100)}"
            method_name = f"Ret.{int(level*100)}%"
            if key in all_results["seeds"][str(config.seeds[0])]["graded"][level_key]:
                vals = [all_results["seeds"][str(s)]["graded"][level_key][key] for s in config.seeds]
                summary_stats[metric][method_name] = {"mean": np.mean(vals), "std": np.std(vals)}

    save_summary_table(config.results_dir, summary_stats, all_methods)

    # Print summary
    logger.info("\nSummary Table:")
    logger.info(f"{'Metric':<15}" + "".join([f"{m:<22}" for m in all_methods]))
    logger.info("-" * (15 + 22 * len(all_methods)))
    for metric, _ in SUMMARY_METRICS:
        row = f"{metric:<15}"
        for method in all_methods:
            if method in summary_stats[metric]:
                m = summary_stats[metric][method]["mean"]
                s = summary_stats[metric][method]["std"]
                row += f"{m:.5f}±{s:.5f}".ljust(22)
            else:
                row += "N/A".ljust(22)
        logger.info(row)

    # Save final results
    final_path = os.path.join(config.results_dir, f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {final_path}")
    logger.info(f"Output folder: {config.results_dir}")

    return all_results


if __name__ == "__main__":
    # Full configuration: 2400 samples, 5 seeds with disjoint forget sets, GPU
    # Uses forget05 pool (2000 rows) partitioned into 5×400 = completely disjoint
    config = TOFURetrainConfig(
        max_forget_samples=400,
        max_retain_samples=1000,
        max_test_samples=1000,
        seeds=[1, 2, 3, 4, 5],
        disjoint_forget=True,
        num_epochs=3,
        batch_size=4,
        device="cuda",
    )

    print("Starting TOFU true retraining MIAU experiment...")
    print(f"Total samples: {config.max_forget_samples + config.max_retain_samples + config.max_test_samples}")
    print(f"Seeds: {config.seeds}")
    print(f"Device: {config.device}")
    print()

    results = run_experiment(config)
