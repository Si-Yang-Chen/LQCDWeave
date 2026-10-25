from typing import Literal, NamedTuple

# Internal annotation for one quark line.  The public ``Flavor`` name is the
# symbolic HadronFlavorStructure alias exported by operators.correlator.api.
QuarkFlavor = Literal["u", "d", "s", "c", "t", "b"]


class Tag(NamedTuple):
    tag: int
    time: int | str

    def __eq__(self, other):
        return self.tag == other.tag and self.time == other.time


__all__ = ["Tag"]
