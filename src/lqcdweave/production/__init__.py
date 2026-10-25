"""Public production facade; independent modules load only when requested."""
from importlib import import_module

_PUBLIC = {
    "NoisevectorGenerator": "lqcdweave.production.noisevector.api",
    "PerambulatorGenerator": "lqcdweave.production.perambulator.api",
    "LaplaceEigenvectorGenerator": "lqcdweave.production.eigenvector.api",
    "ElementalGenerator": "lqcdweave.production.elemental.api",
    "generate_baryon_elemental": "lqcdweave.production.elemental.api",
    "BaryonElementalGenerator": "lqcdweave.production.elemental.api",
}

def __getattr__(name):
    try:
        module = _PUBLIC[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value

__all__ = ['NoisevectorGenerator', 'PerambulatorGenerator', 'LaplaceEigenvectorGenerator', 'ElementalGenerator', 'generate_baryon_elemental', 'BaryonElementalGenerator']
