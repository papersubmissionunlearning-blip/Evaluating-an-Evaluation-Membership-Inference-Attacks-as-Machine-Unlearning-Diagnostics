"""Minimal stand-in for upstream `ImageClassification/utils.py`.

Upstream's utils imports its whole data and model stack (`from dataset import *`,
`from models import *`, `from imagenet import prepare_data`), none of which is
used here -- the host project supplies the data and the model. Vendoring it
wholesale would also shadow the host's own `models` / `datasets` modules.

`warmup_lr`, `save_checkpoint`, `load_checkpoint`, `AverageMeter` and `accuracy`
below are copied verbatim from upstream utils.py (lines 35-83 and 505-518).
`setup_model` delegates to a factory registered by the host.
"""
import os
import shutil

import torch


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ---------------------------------------------------------------------------
# Host integration
# ---------------------------------------------------------------------------
# `unlearn/impl.py::_iterative_scrub_impl` calls `utils.setup_model(args)` to
# build the frozen teacher for SCRUB, then loads the student's state dict into
# it. The teacher must therefore be the host project's architecture, not
# upstream's -- otherwise load_state_dict fails or distills from the wrong net.

_MODEL_FACTORY = None


def register_model_factory(factory):
    """Register a callable `factory(args) -> nn.Module` used by setup_model."""
    global _MODEL_FACTORY
    _MODEL_FACTORY = factory


def setup_model(args, rank=None):
    if _MODEL_FACTORY is None:
        raise RuntimeError(
            "No model factory registered. Call "
            "dualoptim.utils.register_model_factory(...) before running SCRUB; "
            "dualoptim_adapter.build_args() does this for you."
        )
    return _MODEL_FACTORY(args)
