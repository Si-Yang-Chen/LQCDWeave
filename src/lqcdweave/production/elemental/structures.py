"""Spatial recipes expressed through transport, not separate generators."""

from itertools import product

from lqcdweave.contracts.elemental import GaugePath, integer
from .transport import apply_transport, path_transport


def paths_up_to(order, version="legacy"):
    order = integer(order, "num_nabla", 0)
    if version not in ("legacy", "v2"):
        raise ValueError("path_version must be legacy or v2")
    paths = [GaugePath()]
    frontier = [()]
    for _ in range(order):
        frontier = [p + (d,) for p in frontier for d in range(6 if version == "legacy" else 8)
                    if version == "v2" or not p or abs(p[-1] - d) != 3]
        paths.extend(GaugePath(p) for p in frontier)
    return tuple(paths)


def derivatives_up_to(order):
    return tuple(p for n in range(integer(order, "num_nabla", 0) + 1) for p in product(range(3), repeat=n))


def derivative_assignments(directions):
    """Preserve ordered left/right derivative action and its sign."""
    for mask in range(1 << len(directions)):
        right = tuple(d for i, d in enumerate(directions) if mask & (1 << i))
        left = tuple(d for i, d in enumerate(directions) if not mask & (1 << i))[::-1]
        yield left, right, (-1) ** len(right)


def covariant_derivative(vectors, direction, links, t, xp):
    """Apply one spatial difference using source links in their native axes."""
    link=links[direction,t]
    forward=xp.roll(vectors,-1,axis=3-direction)
    forward=xp.einsum("zyxab,ezyxb->ezyxa",link,forward,optimize=True)
    backward=xp.einsum("zyxba,ezyxb->ezyxa",link.conj(),vectors,optimize=True)
    backward=xp.roll(backward,1,axis=3-direction)
    return forward-backward


def differentiate(vectors, directions, links, t, xp, step_transports=None):
    result = vectors
    for d in directions:
        result=covariant_derivative(result,d,links,t,xp)
    return result


def displace(vectors, distance, links, t, xp):
    if distance == 0:
        return vectors
    result = xp.zeros_like(vectors, dtype=complex)
    for d in range(6):
        path = GaugePath((d,) * distance)
        result += apply_transport(path_transport(links, path, t, array_module=xp), vectors,
                                  path.displacement, array_module=xp) / 6
    return result
