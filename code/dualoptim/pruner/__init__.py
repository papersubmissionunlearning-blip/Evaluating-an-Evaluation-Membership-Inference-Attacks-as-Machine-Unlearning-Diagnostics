"""Trimmed version of upstream `pruner/__init__.py`.

Upstream also re-exports `omp` and `synflow`, which transitively import the
`trainer` package. Neither is referenced by the vendored unlearning code --
the only pruner symbols used are `extract_mask`, `prune_model_custom`,
`remove_prune` and `check_sparsity`, all of which live in `utils.py` and
depend on nothing beyond torch. The two unused modules are therefore not
vendored.
"""
from .utils import *
