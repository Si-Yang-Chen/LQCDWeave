"""Lower proven complete spatial diagonals without constraining spin."""

from dataclasses import replace
from lqcdweave.contracts.elemental_sectors import sector_layout

from lqcdweave.contracts.correlator.api import AxisPattern


def lower_complete_diagonals(network, factors):
    """Merge endpoint labels and remove the redundant dense vertex axes.

    Sparse P/P records retain their existing bridge representation. A proof
    belongs to each selected factor, so mixed operator sums remain valid.
    """
    parent, reduced = {}, {}

    def find(label):
        parent.setdefault(label, label)
        if parent[label] != label:
            parent[label] = find(parent[label])
        return parent[label]

    for index, request in enumerate(network.requests):
        if request.kind != "vertex":
            continue
        metadata = factors[request.vertex_index].tensor
        if metadata.structure != "diagonal" or metadata.sparse_layout != "dense":
            continue
        labels = request.subscripts
        pairs = ((2, 3),) if sector_layout(request.sector) == "v2v" else ((2, 4), (3, 5))
        for left, right in pairs:
            parent[find(labels[right])] = find(labels[left])
        reduced[index] = tuple(axis for axis in range(len(labels))
                               if axis not in {right for _left, right in pairs})
    if not reduced:
        return network
    requests = tuple(replace(request, subscripts=tuple(
        find(request.subscripts[axis])
        for axis in reduced.get(index, range(len(request.subscripts)))))
        for index, request in enumerate(network.requests))
    return replace(network, requests=requests,
                   equation=AxisPattern(tuple(item.subscripts for item in requests), ()))
