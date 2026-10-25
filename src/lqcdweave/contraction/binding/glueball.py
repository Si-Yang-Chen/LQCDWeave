"""Bind a zero-quark insertion to an already measured component."""

from copy import copy
import numpy as np


class Glueball:
    """Select one component and momentum; source conjugation is applied once."""

    def __init__(self, elemental, component, momentum=0, source=False, *,
                 name=None, array_module=np):
        self.elemental = elemental
        self.reader = elemental if callable(getattr(elemental, "block", None)) else None
        self.component, self.momentum = component, momentum
        self.dagger = bool(source)
        self.name = name
        self.xp = array_module
        self.key = getattr(self.reader, "cfg", None)

    def with_reader(self, reader, *, configuration=None):
        if not callable(getattr(reader, "describe_block", None)) or not callable(
                getattr(reader, "block", None)):
            raise TypeError("Glueball requires a reader with describe_block and block")
        cfg = getattr(reader, "cfg", None)
        if configuration is not None and cfg is not None and configuration != cfg:
            raise ValueError("configuration disagrees with glueball reader")
        result = copy(self)
        result.reader = reader
        result.key = configuration if configuration is not None else cfg
        return result

    def load(self, key):
        reader = self.reader if self.reader is not None else self.elemental.load(key)
        bound = self.with_reader(reader, configuration=key)
        self.reader, self.key = bound.reader, bound.key
        return self


__all__ = ["Glueball"]
