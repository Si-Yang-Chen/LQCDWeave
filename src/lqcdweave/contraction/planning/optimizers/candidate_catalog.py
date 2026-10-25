"""Bounded computation-layout catalog with atomic metadata-only updates."""
from dataclasses import dataclass
from itertools import combinations

from ..contraction_index import ContractionKey
from ..logical_index import bridge_identity, join_identity


class CatalogOverflow(Exception):
    """Use full enumeration when a lookup cache exceeds its internal cap."""


@dataclass(frozen=True, slots=True)
class CandidateRow:
    term: object
    specs: tuple
    joins: tuple
    entries: tuple


class CandidateCatalog:
    """Keep first ordered occurrence per Term and computation, never states.

    Complete algebra keys resolve hash collisions. Retarget hints are used
    only when the index's previous context is this catalog's exact context;
    otherwise every row's expression, input specs and join constraints are
    checked. Interrupted updates publish nothing.
    """
    capacity = 4096

    def __init__(self, problem):
        self.problem = problem
        self.rows, self.groups = {}, {}
        self.context = None
        self.count = 0
        self.row_builds = self.row_reuses = 0

    def layouts(self, state, index, checkpoint):
        if state.problem is not self.problem:
            raise ValueError("candidate catalog belongs to another problem")
        if len(state.pending) > self.capacity:
            raise CatalogOverflow()
        rows, groups, edited = dict(self.rows), dict(self.groups), set()
        count, builds, reuses = self.count, 0, 0

        def edit(identity):
            if identity not in edited:
                groups[identity] = dict(groups.get(identity, {}))
                edited.add(identity)
            return groups.setdefault(identity, {})

        previous = index.previous_context
        context = self.context
        if (context is not None and previous is not None and context[0] is previous[0]
                and context[1] is previous[1] and context[2] == previous[2]):
            indices = sorted(index.changed_rows)
            reuses = len(state.pending) - len(indices)
        else:
            indices = range(len(state.pending))
        for i in indices:
            checkpoint()
            term = state.pending[i]
            specs = tuple(state.catalog[f.tensor] for f in term.factors) if term else ()
            joins = tuple(join_identity(j) for j in index.scene_joins.get(i, ()))
            old = rows.get(i)
            if old is not None and old.term is term and old.specs == specs and old.joins == joins:
                reuses += 1
                continue
            if old is not None:
                for identity, pair, layout in old.entries:
                    checkpoint()
                    group = edit(identity)
                    del group[i]
                    if not group:
                        del groups[identity]
                count -= len(old.entries)
            entries, seen = [], set()
            if term is not None:
                pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
                for pair in pairs:
                    checkpoint()
                    try:
                        layout = index.layout(i, pair)
                    except ValueError:
                        continue
                    identity = ContractionKey(("contract", layout[0], layout[1],
                                               bridge_identity(layout[5]), bridge_identity(layout[6])))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    # The active join table changes on retarget; retain only
                    # immutable local layout data and bind the current table.
                    layout = layout[:-1]
                    entries.append((identity, pair, layout))
                    edit(identity)[i] = pair, layout
                    count += 1
                    if count > self.capacity:
                        raise CatalogOverflow()
            rows[i] = CandidateRow(term, specs, joins, tuple(entries))
            builds += 1
        representatives = []
        for identity, group in groups.items():
            checkpoint()
            i = min(group)
            pair, layout = group[i]
            representatives.append((i, pair, layout + (index.scene_joins,), identity))
        representatives.sort(key=lambda value: value[:2])
        checkpoint()
        # Publish only after all metadata work, ordering and budget checks.
        self.rows, self.groups, self.count = rows, groups, count
        self.context = state.pending, state.catalog, state.completed_joins
        self.row_builds += builds
        self.row_reuses += reuses
        return tuple(representatives)
