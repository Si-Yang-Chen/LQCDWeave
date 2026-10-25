"""Lazy public optimizer exports.

Search implementations import transition and scoring helpers.  Keeping this
package initializer lazy prevents a storage -> actions -> scoring import from
loading every optimizer back into the partially initialized storage module.
"""

from importlib import import_module

_EXPORTS = {
    "solve_storage": ".session",
    "SearchSpace": ".protocols",
    "LayeredSearchSpace": ".protocols",
    "solve_shortest_path": ".search",
    "solve_layered_search": ".search",
    "StorageSearchSpace": ".storage_space",
    "solve_storage_exact": ".storage_space",
    "StorageSearchResult": ".results",
    "verify_route": ".verification",
    "VerificationError": ".verification",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
