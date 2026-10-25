"""Lazy shapes for retired table APIs, derived from the public implementation."""
from lazy_object_proxy import Proxy


def row_connections():
    from .api import irrep_matrices, row_transfer
    from .hardcoded_rep import OD_irreps, little_group_irreps
    return {p: {name: [list(row_transfer(tuple(map(int, p.split(','))), name, r))
                       for r in range(irrep_matrices(tuple(map(int, p.split(','))), name)['iden'].rows)]
                for name in (OD_irreps if p == '0,0,0' else entries)}
            for p, entries in little_group_irreps.items()}


def reductions(momentum):
    from .hardcoded_rep import OD_irreps, little_group_irreps
    from .api import restrict_irrep

    def source_entries(source):
        result = {}
        for target in little_group_irreps[','.join(map(str, momentum))]:
            reduction = restrict_irrep(momentum, source, target)
            if reduction['multiplicity']:
                result[target] = [[list(row[:, copy]) for row in reduction['row_bases']]
                                  for copy in range(reduction['multiplicity'])]
        return result

    return {name + parity: Proxy(lambda source=name + parity: source_entries(source))
            for name in OD_irreps for parity in ('g', 'u')}
