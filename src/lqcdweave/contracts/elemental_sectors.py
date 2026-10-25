"""Spatial endpoint names; V and D deliberately share numerical layouts."""

ELEMENTAL_SECTORS = tuple(f"{left}2{right}" for left in "vdp" for right in "vdp")


def sector_kinds(sector):
    if sector not in ELEMENTAL_SECTORS:
        raise ValueError("unknown elemental sector")
    return sector[0], sector[2]


def reverse_sector(sector):
    left, right = sector_kinds(sector)
    return f"{right}2{left}"


def sector_layout(sector):
    left, right = sector_kinds(sector)
    return f"{'p' if left == 'p' else 'v'}2{'p' if right == 'p' else 'v'}"


def elemental_tail_axes(sector):
    left, right = sector_kinds(sector)
    return (("point_left", "color_left") if left == "p" else ("mode_left",)) + (
        ("point_right", "color_right") if right == "p" else ("mode_right",)
    )


def propagator_axes(sector):
    sector = sector.lower()
    if len(sector) != 3 or sector[1] != "s" or sector[0] not in "vdp" or sector[2] not in "vdp":
        raise ValueError("unknown propagator sector")
    def endpoint(kind, side):
        if kind == "p":
            return side + "_point", side + "_color"
        return (side + ("_dilution_vector" if kind == "d" else "_mode"),)
    return ("sink_spin", "source_spin") + endpoint(sector[0], "sink") + endpoint(sector[2], "source")


__all__ = ["ELEMENTAL_SECTORS", "sector_kinds", "reverse_sector", "sector_layout",
           "elemental_tail_axes", "propagator_axes"]
