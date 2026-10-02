"""Vendored subset of the official DualOptim image-classification code.

Upstream: https://github.com/CityU-MLO/DualOptim  (ImageClassification/)
Paper:    Zhong, Luo & Liu, "DualOptim: Enhancing Efficacy and Stability in
          Machine Unlearning with Dual Optimizers", NeurIPS 2025.

The algorithm bodies in `unlearn/` (SCRUB, RL/SalUn, SFRon) and the optimizer
construction in `unlearn/impl.py` are byte-identical to upstream. The only
edits are import statements, which are re-pointed at this package so that
upstream's bare `import utils` / `import pruner` / `from optim import ...`
do not resolve against the host project's same-named top-level modules.

`utils.py` here is a shim: it carries verbatim copies of the four helpers the
algorithms actually call, and delegates `setup_model` to the host project so
SCRUB's teacher is built from the host architecture.

Note: upstream hard-codes `.cuda()` in the training loops. Those calls are
left untouched to preserve exactness, so a CUDA device is required.
"""
