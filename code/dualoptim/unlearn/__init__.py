"""Trimmed version of upstream `unlearn/__init__.py`.

Upstream registers ~20 methods (GA, FT, fisher, Wfisher, boundary_*, prune
variants, ...). Only the three this project runs are vendored, so only those
three are registered here; `get_unlearn_method` keeps upstream's signature and
raises the same NotImplementedError for anything else.
"""
from .RL import RL
from .scrub import SCRUB
from .sfron import SFRon
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "RL":
        return RL
    elif name == "SCRUB":
        return SCRUB
    elif name == "SFRon":
        return SFRon
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
