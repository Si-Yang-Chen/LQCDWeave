"""Conservative operator support proofs, independent of numerical values."""


def elemental_structures(rows, momenta, blocks, *, orthogonal_basis=False, orthogonal_bases=None,
                         point_diagonal_components=()):
    """Describe only proven non-general component/momentum/sector entries."""
    result = {}
    point_components = set(point_diagonal_components)
    orthogonal_bases = dict(orthogonal_bases or {})
    orthogonal_bases.setdefault("V", orthogonal_basis)
    for component, terms in rows:
        spatial_identity = all(not path.steps for path, _coefficient in terms)
        for momentum_index, momentum in enumerate(momenta):
            for kind, proven in orthogonal_bases.items():
                sector = f"{kind.lower()}2{kind.lower()}"
                if (sector in blocks and spatial_identity and proven
                        and tuple(momentum) == (0, 0, 0)):
                    result[sector, component, momentum_index] = "diagonal"
            if "p2p" in blocks and component in point_components:
                result["p2p", component, momentum_index] = (
                    "diagonal" if spatial_identity else "point_diagonal")
    return result
