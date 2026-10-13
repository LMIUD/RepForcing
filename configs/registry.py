# Modified for the RepForcing source release; see NOTICE.md.
"""Variant lookup by `"<module>.<variant>"` name.

    from configs.registry import get_config
    config = get_config("dmd.framewise")

The registry is built from configuration modules at import time and is read-only.
All configurations use train.py, which selects the trainer through config.trainer.
"""

from . import dmd, rft


_MODULES = {
    "dmd": dmd,
    "rft": rft,
}


def _variants(module):
    """Public zero-argument factories declared in a variant module."""
    return {
        name: fn
        for name, fn in vars(module).items()
        if not name.startswith("_") and callable(fn) and fn.__module__ == module.__name__
    }


_REGISTRY = {
    f"{ns}.{name}": fn
    for ns, module in _MODULES.items()
    for name, fn in _variants(module).items()
}


def available():
    """Every registered variant name, sorted."""
    return sorted(_REGISTRY)


def get_config(name):
    """Build the config for `name`. Raises KeyError listing all names if unknown."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown config variant {name!r}. Available variants:\n  "
            + "\n  ".join(available())
        ) from None
    return factory()
