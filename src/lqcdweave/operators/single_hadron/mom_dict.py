"""Legacy momentum tables, loaded from ``presets/momentum/*.json``.

The tables are data: their ordering is an identity of already-produced
elemental data, so the JSON documents are the single source and this module
only exposes them under their historical names. New code should pass
``momentum_recipe(...).momenta`` (ordered XYZ triples) explicitly instead of a
slot-numbered mapping.
"""

from collections.abc import Mapping

from lqcdweave.contracts.elemental import normalize_momenta

from .recipes import momentum_recipe


def _table(name):
    return momentum_recipe(name).as_mapping()


momDict_mom1 = _table("mom1")
momDict_mom2 = _table("mom2")
momDict_mom3 = _table("mom3")
momDict_mom9 = _table("mom9")
momDict_test = _table("mom_test")


def mom_dict_to_list(mom):
    """Return ordered XYZ values from a slot->"x y z" table (keys are not slots)."""
    if not isinstance(mom, Mapping):
        raise TypeError("pass a momentum mapping; preset names live in momentum_recipe()")
    return list(normalize_momenta(mom))