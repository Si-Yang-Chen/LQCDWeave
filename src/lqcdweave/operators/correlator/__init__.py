"""Public data-independent hadron symbols and Wick graph operations."""

try:
    from .api import *  # noqa: F401,F403
    from .api import __all__
except ModuleNotFoundError as exc:
    if exc.name not in {"sympy", "scipy", "lazy_object_proxy"}:
        raise
    raise ModuleNotFoundError(
        f"Missing optional dependency {exc.name!r}. Optional Correlator support requires "
        "Python 3.11+ and the [correlator] extra. "
        "Run python -m pip install -e '.[correlator]'. "
        "See docs/modules/correlator/correlator_frontend.md.",
        name=exc.name,
    ) from exc
