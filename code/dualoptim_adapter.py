"""Bridge between this project's unlearning pipeline and the vendored DualOptim code.

The vendored algorithms in `dualoptim/unlearn/` are byte-identical to upstream,
which means they expect upstream's conventions:

  * batches are 2-tuples ``(image, target)``; this project's loaders yield
    3-tuples ``(image, _, target)``
  * forget/retain loaders arrive as a dict ``{"forget": ..., "retain": ...}``
  * ~25 hyperparameters are read off an argparse ``Namespace``
  * ``utils.setup_model(args)`` builds SCRUB's teacher
  * ``args.dataset`` is compared against lowercase literals such as ``"cifar10"``

Everything in this module exists to satisfy those expectations without editing
the algorithms themselves.

Hyperparameters come from Table 7 of the DualOptim paper (Appendix B.1),
"Summary of hyperparameters for each method on unlearning 10% random subset of
CIFAR-10" -- which is exactly this project's CIFAR-10 ResNet-18 setting
(forget_per_class=500 x 10 classes = 5000 = 10% of CIFAR-10).

Note: upstream hard-codes ``.cuda()`` inside the training loops. Those calls
are deliberately left untouched, so these methods require a CUDA device.
"""
import os
import shutil
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

import models
from dualoptim import utils as do_utils
from dualoptim.generate_mask import save_gradient_ratio
from dualoptim.unlearn import get_unlearn_method


# ---------------------------------------------------------------------------
# Paper hyperparameters -- DualOptim (NeurIPS 2025), Appendix B, Table 7.
# 10% random subset of CIFAR-10, ResNet-18.
#
# "Adam (F) + Adam (R)" is upstream's `dual_aa`; "Adam (F) + SGD (R)" is `dual`.
# Baseline SCRUB is listed as plain "Adam", i.e. the single-optimizer branch.
# ---------------------------------------------------------------------------
PAPER_HPARAMS = {
    # name          unlearn     optim      eta_f    eta_r   extra
    "scrub":    dict(unlearn="SCRUB", optim="adam",    unlearn_lr=1e-4,   retain_lr=1e-4, unlearn_epochs=10),
    "salun":    dict(unlearn="RL",    optim="sgd",     unlearn_lr=0.018,  retain_lr=0.01, unlearn_epochs=10, use_mask=True),
    "sfron":    dict(unlearn="SFRon", optim="sgd",     unlearn_lr=0.01,   retain_lr=0.01, sfron_iters=1500, sfron_freq=5, sfron_alpha=31),
    "scrub_do": dict(unlearn="SCRUB", optim="dual_aa", unlearn_lr=1.5e-4, retain_lr=1e-4, unlearn_epochs=10),
    "salun_do": dict(unlearn="RL",    optim="dual",    unlearn_lr=1.3e-4, retain_lr=0.01, unlearn_epochs=10, use_mask=True),
    "sfron_do": dict(unlearn="SFRon", optim="dual",    unlearn_lr=1e-4,   retain_lr=0.01, sfron_iters=1500, sfron_freq=5, sfron_alpha=1),
}

# Appendix B.1: "all baselines use the SGD optimizer with momentum of 0.9,
# weight decay of 5e-4, and batch size of 128".
PAPER_MOMENTUM = 0.9
PAPER_WEIGHT_DECAY = 5e-4
PAPER_BATCH_SIZE = 128

# The paper's SalUn row uses the `with_0.5.pt` saliency mask (README example).
SALUN_MASK_THRESHOLD = 0.5

# Upstream's `--dataset` values are lowercase. `RL.py` branches on
# `args.dataset == "cifar10"`; passing "Cifar10" would silently route SalUn
# down the CIFAR-100 code path, so host dataset names are normalised here.
_DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar20": "cifar100",   # Cifar20 is the 20-superclass view of CIFAR-100
    "cifar100": "cifar100",
    "mnist": "cifar10",      # single-pass forget/retain branch
    "svhn": "svhn",
    "mucac": "cifar10",
}


def normalise_dataset_name(dataset_name):
    return _DATASET_ALIASES.get(str(dataset_name).lower(), str(dataset_name).lower())


# ---------------------------------------------------------------------------
# Loader adaptation
# ---------------------------------------------------------------------------
class TwoTupleLoader:
    """Present a 3-tuple loader as the 2-tuple loader upstream expects.

    Upstream iterates ``for image, target in loader`` and also reaches for
    ``loader.dataset`` (``RL`` deep-copies it) and ``len(loader)`` (used to
    average gradients and to size the progress output), so both are forwarded.
    """

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if len(batch) == 3:
                image, _, target = batch
            else:
                image, target = batch
            yield image, target

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset

    @property
    def batch_size(self):
        return self._loader.batch_size

    def __getattr__(self, item):
        return getattr(self._loader, item)


def make_data_loaders(retain_train_dl, forget_train_dl):
    """Build the ``{"forget": ..., "retain": ...}`` dict upstream expects."""
    return {
        "forget": TwoTupleLoader(forget_train_dl),
        "retain": TwoTupleLoader(retain_train_dl),
    }


# ---------------------------------------------------------------------------
# args shim
# ---------------------------------------------------------------------------
def build_args(
    method,
    dataset_name,
    model_name,
    num_classes,
    seed,
    save_dir,
    img_size=32,
    overrides=None,
):
    """Assemble the Namespace the vendored algorithms read.

    Defaults match upstream `arg_parser.py`; learning rates and optimizer
    choice come from `PAPER_HPARAMS`. Anything in `overrides` wins, so a
    caller can deviate from the paper deliberately.
    """
    if method not in PAPER_HPARAMS:
        raise ValueError(
            f"Unknown DualOptim method {method!r}; expected one of "
            f"{sorted(PAPER_HPARAMS)}"
        )
    hp = dict(PAPER_HPARAMS[method])
    hp.pop("use_mask", None)

    args = SimpleNamespace(
        # --- from the paper / upstream defaults ---
        momentum=PAPER_MOMENTUM,
        weight_decay=PAPER_WEIGHT_DECAY,
        batch_size=PAPER_BATCH_SIZE,
        # Upstream default. With unlearn_epochs=10 these milestones are never
        # reached, which is how the paper's "constant scheduler" is realised.
        decreasing_lr="91,136",
        warmup=0,
        print_freq=50,
        rewind_epoch=0,
        rewind_pth=None,
        imagenet_arch=False,
        gpu=0,
        input_size=img_size,
        # upstream arg_parser defaults; the paper does not override these
        scrub_alpha=0.5,
        scrub_gamma=1.0,
        sfron_alpha=25,
        sfron_freq=5,
        sfron_iters=1500,
        unlearn_epochs=10,
        # --- run-specific ---
        num_classes=num_classes,
        dataset=normalise_dataset_name(dataset_name),
        seed=seed,
        save_dir=save_dir,
        # `warmup_lr` reads args.lr; unreachable at warmup=0 but kept coherent
        lr=hp["unlearn_lr"],
        # host-only fields, used by the model factory below
        arch=model_name,
    )
    for key, value in hp.items():
        setattr(args, key, value)
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)

    # `iterative_scrub` calls utils.setup_model(args) to build SCRUB's frozen
    # teacher and then loads the student's state dict into it, so the teacher
    # must be the host architecture.
    do_utils.register_model_factory(
        lambda a: getattr(models, a.arch)(num_classes=a.num_classes)
    )
    return args


# ---------------------------------------------------------------------------
# SalUn saliency mask
# ---------------------------------------------------------------------------
def salun_mask_dir(dataset_name, model_name, seed):
    """Scratch location for a seed's saliency mask.

    Keyed by (dataset, model, seed) because the mask depends on the baseline
    weights and on the seed's forget/retain split -- but not on the learning
    rate, so `salun` and `salun_do` of the same seed reuse one mask.
    """
    return os.path.join(
        tempfile.gettempdir(),
        "dualoptim_masks",
        f"{dataset_name}_{model_name}_seed{seed}",
    )


def get_salun_mask(model, data_loaders, args, mask_dir, threshold=SALUN_MASK_THRESHOLD):
    """Return SalUn's saliency mask, generating it on first use.

    Masks are cached per (model, dataset, seed) so the `salun` and `salun_do`
    runs of a given seed share one mask rather than recomputing it.
    """
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"with_{threshold}.pt")

    if not os.path.exists(mask_path):
        print(f"[dualoptim] generating SalUn saliency mask -> {mask_path}")
        _prune_other_mask_dirs(mask_dir)
        mask_args = SimpleNamespace(**vars(args))
        mask_args.save_dir = mask_dir
        # Upstream generates the mask with SGD at the unlearning LR.
        save_gradient_ratio(data_loaders, model, nn.CrossEntropyLoss(), mask_args)
        # `save_gradient_ratio` is vendored verbatim, so it writes a mask for
        # every threshold in {0.1 ... 1.0}. Each is the size of the model's
        # parameters (~90 MB for ResNet-18), so drop the nine we don't use
        # instead of leaving ~900 MB per seed behind.
        _keep_only(mask_dir, os.path.basename(mask_path))
    else:
        print(f"[dualoptim] reusing cached SalUn mask {mask_path}")

    return torch.load(mask_path, map_location="cpu")


def _keep_only(directory, filename):
    for entry in os.listdir(directory):
        if entry != filename:
            try:
                os.remove(os.path.join(directory, entry))
            except OSError:
                pass


def _prune_other_mask_dirs(keep_dir):
    """Drop cached masks for other seeds; only the active seed's is needed."""
    parent = os.path.dirname(keep_dir)
    if not os.path.isdir(parent):
        return
    for entry in os.listdir(parent):
        path = os.path.join(parent, entry)
        if os.path.isdir(path) and os.path.abspath(path) != os.path.abspath(keep_dir):
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_unlearn(method, model, retain_train_dl, forget_train_dl, args, mask=None):
    """Run one vendored unlearning algorithm; returns the unlearned model.

    SCRUB and RL are `iterative_unlearn`-decorated functions that mutate the
    model in place. SFRon is a class driven the way `main_random_sfron.py`
    drives it, and its `get_unlearned_model()` returns a 19-tuple whose first
    element is the model.
    """
    data_loaders = make_data_loaders(retain_train_dl, forget_train_dl)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.save_dir, exist_ok=True)

    algorithm = get_unlearn_method(args.unlearn)

    if args.unlearn == "SFRon":
        unlearn_method = algorithm(model, criterion, args.save_dir, args)
        unlearn_method.prepare_unlearn(data_loaders)
        result = unlearn_method.get_unlearned_model()
        # upstream unpacks 19 values, model first
        return result[0] if isinstance(result, tuple) else result

    # SCRUB / RL: mutates `model` in place
    algorithm(data_loaders, model, criterion, args, mask)
    return model


class scratch_dir:
    """Temp directory for upstream's per-epoch checkpoints.

    `impl.py` unconditionally writes `checkpoint_{epoch}.pt` into
    `args.save_dir` every epoch. Those are training artefacts, not results, so
    they are routed to a scratch directory and removed afterwards rather than
    accumulating next to the experiment outputs.
    """

    def __init__(self, keep=False):
        self.keep = keep
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="dualoptim_")
        return self.path

    def __exit__(self, exc_type, exc, tb):
        if self.path and not self.keep:
            shutil.rmtree(self.path, ignore_errors=True)
        return False
