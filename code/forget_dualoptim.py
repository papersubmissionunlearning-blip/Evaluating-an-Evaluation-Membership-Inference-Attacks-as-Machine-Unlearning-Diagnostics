"""The six DualOptim unlearning methods, exposed with this project's method signature.

Methods: scrub, salun, sfron, scrub_do, salun_do, sfron_do
(`_do` = the DualOptim variant, matching the paper's "+DO" shorthand.)

The unlearning itself is performed by the vendored, byte-identical upstream code
in `dualoptim/`; this module only converts between the two calling conventions
via `dualoptim_adapter`.

`get_metric_scores` is imported from `forget.py` rather than copied, so these
six methods are scored by exactly the same code as the existing nine methods.
That is what makes the new CSV rows comparable to the existing ones -- a
divergent copy of the metric code would silently break that.
"""
import torch
import torch.nn as nn

from forget import get_metric_scores

import dualoptim_adapter as doa


def _run_dualoptim_method(
    method,
    model,
    seed,
    retain_train_dl,
    forget_train_dl,
    dataset_name,
    model_name,
    num_classes,
    device,
    img_size=32,
    overrides=None,
):
    """Shared plumbing for all six methods; returns the unlearned model."""
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"DualOptim method {method!r} requires a CUDA device: the vendored "
            "upstream training loops call .cuda() directly and were left "
            "unmodified to keep them identical to the published code."
        )
    model = model.cuda()

    use_mask = doa.PAPER_HPARAMS[method].get("use_mask", False)

    with doa.scratch_dir() as save_dir:
        args = doa.build_args(
            method=method,
            dataset_name=dataset_name,
            model_name=model_name,
            num_classes=num_classes,
            seed=seed,
            save_dir=save_dir,
            img_size=img_size,
            overrides=overrides,
        )

        print(
            f"[dualoptim] {method}: unlearn={args.unlearn} optim={args.optim} "
            f"lr_f={args.unlearn_lr} lr_r={args.retain_lr}"
        )

        mask = None
        if use_mask:
            # The saliency mask is read off accumulated gradients; upstream
            # never calls optimizer.step() during generation, so the mask does
            # not depend on the learning rate and one mask per (model, data,
            # seed) is shared by `salun` and `salun_do`.
            mask_dir = doa.salun_mask_dir(dataset_name, model_name, seed)
            loaders = doa.make_data_loaders(retain_train_dl, forget_train_dl)
            mask = doa.get_salun_mask(model, loaders, args, mask_dir)
            mask = {k: v.cuda() for k, v in mask.items()}

        model = doa.run_unlearn(
            method, model, retain_train_dl, forget_train_dl, args, mask=mask
        )

    return model


def _make_method(method):
    """Build a host-signature unlearning function for one DualOptim method."""

    def _method(
        model,
        seed,
        unlearning_teacher,
        train_dl,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        retain_train_dl_original,
        retain_valid_dl_original,
        forget_train_dl_original,
        forget_valid_dl_original,
        dataset_name,
        model_name,
        device,
        **kwargs,
    ):
        model = _run_dualoptim_method(
            method=method,
            model=model,
            seed=seed,
            retain_train_dl=retain_train_dl,
            forget_train_dl=forget_train_dl,
            dataset_name=dataset_name,
            model_name=model_name,
            num_classes=kwargs.get("num_classes", 10),
            device=device,
            img_size=224 if model_name == "ViT" else (128 if dataset_name == "MUCAC" else 32),
            overrides=kwargs.get("dualoptim_overrides"),
        )

        return get_metric_scores(
            model,
            unlearning_teacher,
            train_dl,
            retain_train_dl,
            retain_valid_dl,
            forget_train_dl,
            forget_valid_dl,
            valid_dl,
            retain_train_dl_original,
            retain_valid_dl_original,
            forget_train_dl_original,
            forget_valid_dl_original,
            device,
            dataset_name,
            model_name,
        )

    _method.__name__ = method
    _method.__qualname__ = method
    _method.__doc__ = (
        f"DualOptim `{method}`: "
        f"unlearn={doa.PAPER_HPARAMS[method]['unlearn']}, "
        f"optim={doa.PAPER_HPARAMS[method]['optim']}, "
        f"lr_f={doa.PAPER_HPARAMS[method]['unlearn_lr']}, "
        f"lr_r={doa.PAPER_HPARAMS[method]['retain_lr']} "
        f"(paper Table 7, CIFAR-10 10% random forget)."
    )
    return _method


scrub = _make_method("scrub")
salun = _make_method("salun")
sfron = _make_method("sfron")
scrub_do = _make_method("scrub_do")
salun_do = _make_method("salun_do")
sfron_do = _make_method("sfron_do")

METHODS = ["scrub", "salun", "sfron", "scrub_do", "salun_do", "sfron_do"]
