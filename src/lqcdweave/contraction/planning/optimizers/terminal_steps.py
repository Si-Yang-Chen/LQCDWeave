"""Dense output layouts and elimination of independent terminal interleavings.

These rules belong to the logical compute domain. Physical storage is scheduled
afterwards. They do not restrict exact search or impose an order on productive
contractions that can compete for factors or sharing.
"""
from dataclasses import replace

from lqcdweave.contracts.correlator.api import Factor, PendingTerm
from ..canonical_operations import describe


def terminal_failure(space, scene, occurrence):
    """One occurrence that cannot end its complete dense expression in place."""
    term = space.terms[scene]
    output = space.registry.patterns[term.pattern].output
    return (occurrence.join is not None or occurrence.cores
            or len(occurrence.positions) != len(term.factors)
            or set(occurrence.output) != set(output))


def is_terminal(space, occurrences):
    """Every occurrence consumes its entire dense expression, up to transpose."""
    for scene, occurrence in occurrences.items():
        if terminal_failure(space, scene, occurrence):
            return False
    return True


def terminal_proofs(space):
    """Non-terminal proofs: class -> the failing consumer scene proving it.

    A class is non-terminal as soon as one occurrence fails. An occurrence and
    its Term only ever change for a scene that is being reindexed, so a proof
    survives until its own scene is dirtied; by_scene is the reverse index that
    drops exactly the invalidated proofs. Anything else that removes a class
    forgets its proof. The table starts over whenever space.groups itself is
    replaced (reset, release, initial publish).
    """
    state = getattr(space, "_terminal_proofs", None)
    if state is None or state[0] is not space.groups:
        state = (space.groups, {}, {})
        space._terminal_proofs = state
    return state


def invalidate_terminal_scenes(space, scenes):
    """Drop proofs whose failing consumer scene is being reindexed."""
    state = getattr(space, "_terminal_proofs", None)
    if state is None or state[0] is not space.groups:
        return
    proven, by_scene = state[1], state[2]
    for scene in scenes:
        found = by_scene.pop(scene, None)
        if found:
            for identity in found:
                proven.pop(identity, None)


def forget_terminal_witness(space, identity):
    """Drop the remembered failure of a class that no longer exists."""
    state = getattr(space, "_terminal_proofs", None)
    if state is None or state[0] is not space.groups:
        return
    proven, by_scene = state[1], state[2]
    scene = proven.pop(identity, None)
    if scene is not None:
        found = by_scene.get(scene)
        found.discard(identity)
        if not found:
            del by_scene[scene]


def group_is_terminal(space, identity):
    """is_terminal for one class, reusing a surviving non-terminal proof."""
    state = terminal_proofs(space)
    proven = state[1]
    if identity in proven:
        return False
    for scene, occurrence in space.groups[identity].items():
        if terminal_failure(space, scene, occurrence):
            proven[identity] = scene
            state[2].setdefault(scene, set()).add(identity)
            return False
    return True


def layout_family_index(space):
    """Layout variants grouped by terminal family.

    remove_layouts must invalidate every variant of an affected family.
    Grouping them once keeps that lookup proportional to the affected families
    instead of the complete variant set, which only ever grows. The index is
    rebuilt whenever space.layout_variants is replaced.
    """
    state = getattr(space, "_layout_family_index", None)
    variants = space.layout_variants
    if state is None or state[0] is not variants:
        families = space.terminal_families
        index = {}
        for identity in variants:
            index.setdefault(families[identity], set()).add(identity)
        state = (variants, index)
        space._layout_family_index = state
    return state[1]


def note_layout_variant(space, variant, family):
    """Record one variant added to space.layout_variants."""
    state = getattr(space, "_layout_family_index", None)
    if state is not None and state[0] is space.layout_variants:
        state[1].setdefault(family, set()).add(variant)


def remove_layouts(space, affected):
    variants = space.layout_variants
    if len(variants) > len(affected):
        # Many variants, few affected families: ask the reverse index first.
        index = layout_family_index(space)
        candidates = set()
        for family in affected:
            found = index.get(family)
            if found:
                candidates.update(found)
    else:
        # Fewer variants than affected families: the single scan is cheaper,
        # but an index that already exists still has to follow the removals.
        families = space.terminal_families
        candidates = {identity for identity in variants if families[identity] in affected}
        state = getattr(space, "_layout_family_index", None)
        index = state[1] if state is not None and state[0] is variants else None
    removed, scenes, dropped = set(), set(), []
    for identity in candidates:
        family = space.terminal_families[identity]
        removed.add(identity)
        dropped.append((family, identity))
        scenes.update(space._remove_group(identity))
        space._remove_terminal(identity)
    for scene in scenes:
        space.rows[scene] = tuple(item for item in space.rows[scene] if item[0] not in removed)
    variants.difference_update(removed)
    if index is not None:
        for family, identity in dropped:
            members = index.get(family)
            if members is not None:
                members.discard(identity)
                if not members:
                    del index[family]
    for _, identity in dropped:
        forget_terminal_witness(space, identity)
    if space.terminal_families:
        for identity in affected:
            space._remove_terminal(identity)


def register_layouts(space, affected, checkpoint):
    """Keep canonical recipes and add only layouts requested by final consumers."""
    identities = sorted((identity for identity in affected if identity in space.groups),
                        key=space.group_ordinals.__getitem__)
    for identity in identities:
        occurrences = space.groups[identity]
        checkpoint()
        if space.debug:
            # Debug re-opens the complete occurrence scan and checks the
            # remembered non-terminal proof against it.
            terminal = is_terminal(space, occurrences)
            if terminal != group_is_terminal(space, identity):
                raise RuntimeError("remembered terminal failure disagrees with the complete scan")
        else:
            terminal = group_is_terminal(space, identity)
        if not terminal:
            continue
        space._set_terminal(identity, identity)
        first = next(iter(occurrences.values()))
        if len(first.positions) != 2:
            continue
        permutations = {}
        for scene, occurrence in occurrences.items():
            output = space.registry.patterns[space.terms[scene].pattern].output
            permutation = tuple(occurrence.output.index(axis) for axis in output)
            if permutation != tuple(range(len(output))):
                permutations.setdefault(permutation, None)
        if not permutations:
            continue
        state = space._state(first.scene)
        for permutation in permutations:
            checkpoint()
            output = tuple(first.output[i] for i in permutation)
            cache_key = (identity, permutation)
            variant = space.registry.layout_recipes.get(cache_key)
            if variant is None:
                action, spec, workspace, *_ = describe(state, 0, first.positions,
                    profile=space.profile, output_order=output)
                variant = space.registry.calculation(action, spec, workspace)
                space.registry.layout_recipes[cache_key] = variant
            # A requested permutation may already be another canonical recipe.
            # That recipe keeps its existing occurrence group and family.
            if variant in space.groups:
                continue
            bases = space.registry.layout_bases.setdefault(variant, [])
            if (identity, permutation) not in bases:
                bases.append((identity, permutation))
            updates = {scene: occurrence._replace(output=tuple(occurrence.output[i] for i in permutation))
                       for scene, occurrence in occurrences.items()}
            space._add_group(variant, updates)
            for scene, occurrence in updates.items():
                space.rows[scene] += ((variant, occurrence),)
            space.layout_variants.add(variant)
            space._set_terminal(variant, identity)
            note_layout_variant(space, variant, identity)
            space.layout_candidates += 1


def completion_cost(space, operation, checkpoint):
    """Cost of one shared recipe and unique required output normalizers."""
    result = space.registry.data[operation.result]
    total, seen = operation.seconds, set()
    for scene, occurrence in space.groups[operation.calculation].items():
        checkpoint()
        output = space.registry.patterns[space.terms[scene].pattern].output
        if occurrence.output == output:
            continue
        # Equivalent consumers need the same one-input normalizer just once.
        permutation = tuple(occurrence.output.index(axis) for axis in output)
        if permutation in seen:
            continue
        seen.add(permutation)
        cache_key = (operation.result, permutation)
        cached = space.registry.normalizer_costs.get(cache_key)
        if cached is not None:
            total += cached
            continue
        state = space._state(scene)
        state = replace(state,
            pending=(PendingTerm((Factor(result.name, occurrence.output),), output),),
            loaded=state.loaded | {result.name},
            catalog=state.catalog.updated(((result.name, result.spec),)))
        action, *_ = describe(state, 0, (0,), profile=space.profile)
        space.registry.normalizer_costs[cache_key] = action.seconds
        total += action.seconds
    return total


def search_candidates(space, checkpoint):
    """Defer pure outputs, then finish one independent family in its best layout."""
    from .integer_space import IntegerOperation
    space.enumerations += 1
    space._index(checkpoint)
    terminal_count = len(space.terminal_families)
    if len(space.groups) > terminal_count:
        for identity in space.groups:
            checkpoint()
            if identity in space.terminal_families:
                continue
            calculation = space.registry.calculations[identity]
            reused = calculation.result in space.available
            seconds = 0.0 if reused else calculation.seconds
            yield IntegerOperation(identity, calculation.result, reused, seconds), seconds
        space.terminal_deferred += terminal_count
        return
    if not terminal_count:
        return
    family = space._terminal_family(checkpoint)
    choices = []
    for identity in sorted(space.terminal_members[family]):
        # Other independent output families cannot enter this ranking. Charge
        # actual layout evaluations rather than enumerating every deferred ID.
        checkpoint()
        calculation = space.registry.calculations[identity]
        reused = calculation.result in space.available
        seconds = 0.0 if reused else calculation.seconds
        choices.append((IntegerOperation(identity, calculation.result, reused, seconds), seconds))
    best = min(choices, key=lambda item: (completion_cost(space, item[0], checkpoint), item[0].calculation))
    space.terminal_resolutions += 1
    space.terminal_deferred += terminal_count - 1
    yield best
