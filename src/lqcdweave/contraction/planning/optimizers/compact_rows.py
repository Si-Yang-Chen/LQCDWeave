"""Weighted identical dense rows, preserving physical-row insertion order."""
import heapq


COMPACT_DENSE_MIN_TERMS = 4096
# Class reconciliation is checked by its caller per rebuilt row; these coarse
# intervals bound the remaining scans so a deadline is still noticed inside one
# oversized pass. Must be a power of two: the scans test it with a bitmask.
REINDEX_CHECKPOINT_INTERVAL = 4096


def _sum_weights(weights, scenes):
    value = 0
    for scene in scenes:
        value += weights[scene]
    return value


def compact_initial_rows(space):
    if (space.debug or space.problem.baryon_joins or space.problem.sparse_bridges or space.has_sum_barriers
            or len(space.initial_terms) < COMPACT_DENSE_MIN_TERMS):
        return None
    classes, representatives, members = {}, [], []
    for scene, term in enumerate(space.initial_terms):
        identity = classes.get(term)
        if identity is None:
            identity = classes[term] = len(representatives)
            representatives.append(scene)
            members.append([])
        members[identity].append(scene)
    # A nearly unique problem cannot amortize the weighted bookkeeping.
    if len(classes) * 4 > len(space.initial_terms) * 3:
        return None
    for name in ('initial_terms', 'contexts', 'symbols', 'join_cores',
                 'bridge_cores', 'join_axes', 'bridge_axes'):
        values = getattr(space, name)
        setattr(space, name, tuple(values[i] for i in representatives))
    return tuple(tuple(group) for group in members)


def reindex(space, checkpoint):
    """Update classes with the exact ordinal events of the expanded row loop.

    A calculation keeps its old ordinal if an unchanged physical consumer
    survives, or a new consumer appears before the last old one is removed.
    Otherwise it is recreated at its first new physical row/pair position.
    """
    from .integer_space import Occurrence
    from .terminal_steps import invalidate_terminal_scenes, remove_layouts, register_layouts

    if not space.rows and not space.groups and space.next_group_ordinal == 0:
        _initial_index(space, checkpoint)
        return

    dirty = space.dirty
    # Every reindexed scene may replace its occurrence and change its Term, so
    # the non-terminal proofs that named it must be re-derived.
    invalidate_terminal_scenes(space, dirty)
    families = space.terminal_families
    current_rows = space.rows
    affected = set()
    mask = REINDEX_CHECKPOINT_INTERVAL - 1
    tick = 0
    for scene in dirty:
        tick += 1
        if tick & mask == 0:
            checkpoint()
        for identity, _ in current_rows.get(scene, ()):
            affected.add(families.get(identity, identity))
    remove_layouts(space, affected)
    old, new, first_ordinal, first_scene, rows, memo = {}, {}, {}, {}, {}, {}
    for scene in sorted(dirty):
        # Rebuilding a row is bounded work; a coarse deadline check keeps the
        # budget contract without one event per candidate metadata bind.
        checkpoint()
        current = current_rows.get(scene, ())
        for identity, _ in current:
            scenes = old.get(identity)
            if scenes is None:
                old[identity] = {scene}
            else:
                scenes.add(scene)
        term = space.terms[scene]
        if term.done:
            rows[scene] = ()
            continue
        space.row_rebuilds += 1
        if term in memo:
            space.row_description_reuses += 1
            row = tuple((identity, Occurrence(scene, occurrence.positions,
                         occurrence.output, None, ())) for identity, occurrence in memo[term])
        else:
            space.row_descriptions += 1
            row = tuple(space._dense_row(scene, checkpoint))
            memo[term] = row
        rows[scene] = row
        # Classes are inserted by their first physical row, so the first scene
        # that contributes an identity is also the one its event must name.
        for ordinal, (identity, occurrence) in enumerate(row):
            added = new.get(identity)
            if added is None:
                new[identity] = {scene: occurrence}
                first_ordinal[identity] = ordinal
                first_scene[identity] = scene
            else:
                added[scene] = occurrence

    weights, firsts, lasts = space._row_weights, space._scene_first, space._scene_last
    group_weights = space._group_weights
    groups = space.groups
    events = []
    event_append = events.append
    tick = 0
    for identity in new:
        tick += 1
        if tick & mask == 0:
            checkpoint()
        previous = groups.get(identity)
        added = new[identity]
        removed = old.get(identity, ())
        persistent = previous is not None and len(previous) > len(removed)
        first = first_scene[identity]
        if previous is None:
            recreate = changed = True
        else:
            if persistent:
                recreate = False
            else:
                # Classes are inserted by their first physical row, so only the
                # last removed consumer can still pre-empt a recreation.
                last = -1
                for scene in removed:
                    extent = lasts[scene]
                    if extent > last:
                        last = extent
                recreate = firsts[first] >= last
            changed = recreate
            if not recreate:
                weight = group_weights[identity]
                if removed != added.keys():
                    weight -= _sum_weights(weights, removed)
                    weight += _sum_weights(weights, added)
                changed = weight != group_weights[identity]
        if previous is not None:
            if recreate:
                space._remove_group(identity)
            else:
                if len(previous) == len(removed):
                    # Every surviving consumer was rebuilt in this pass, so the
                    # new mapping already is the exact content and insertion
                    # order that pop-then-update would produce.
                    groups[identity] = added
                else:
                    for scene in removed:
                        previous.pop(scene, None)
                    previous.update(added)
                group_weights[identity] = weight
        if recreate:
            event_append((firsts[first], first_ordinal[identity], identity))
        # Local scores depend on intrinsic recipe/availability, occurrence
        # weight and ordinal. Position-only updates cannot change them.
        if changed:
            space._touch_group(identity)
    # Identities that lost every consumer arrive here with an empty new row.
    for identity in old:
        if identity in new:
            continue
        tick += 1
        if tick & mask == 0:
            checkpoint()
        previous = groups.get(identity)
        if previous is None:
            space._touch_group(identity)
            continue
        removed = old[identity]
        if len(previous) > len(removed):
            weight = group_weights[identity]
            weight -= _sum_weights(weights, removed)
            changed = weight != group_weights[identity]
            for scene in removed:
                previous.pop(scene, None)
            group_weights[identity] = weight
        else:
            changed = True
            space._remove_group(identity)
        if changed:
            space._touch_group(identity)
    events.sort()
    tick = 0
    for _, _, identity in events:
        tick += 1
        if tick & mask == 0:
            checkpoint()
        space._add_group(identity, new[identity])
    for scene, row in rows.items():
        if row:
            space.rows[scene] = row
        else:
            space.rows.pop(scene, None)
    affected.update(new)
    dirty.clear()
    remove_layouts(space, affected)
    register_layouts(space, affected, checkpoint)
    if len(space.terminal_heap) > max(64, 2 * len(space.terminal_members)):
        heap = []
        for family in space.terminal_members:
            checkpoint()
            heap.append(family)
        heapq.heapify(heap)
        space.terminal_heap = heap


def _initial_index(space, checkpoint):
    """Publish the empty index directly in first-physical-occurrence order.

    No old consumers exist, so there are no removal/recreation events to infer.
    Initial classes are already ordered by their first physical row. Keep all
    mutable index tables local until their complete construction succeeds.
    """
    from .terminal_steps import register_layouts

    rows, groups, ordinals, weights = {}, {}, {}, {}
    for scene in sorted(space.dirty):
        checkpoint()
        if space.terms[scene].done:
            continue
        space.row_rebuilds += 1
        space.row_descriptions += 1
        row = space._dense_row(scene, checkpoint)
        if row:
            rows[scene] = row
        weight = space._row_weights[scene]
        for identity, occurrence in row:
            previous = groups.get(identity)
            if previous is None:
                ordinals[identity] = len(ordinals)
                groups[identity] = {scene: occurrence}
                weights[identity] = weight
            elif scene not in previous:
                previous[scene] = occurrence
                weights[identity] += weight
    space.rows, space.groups = rows, groups
    space.group_ordinals, space.next_group_ordinal = ordinals, len(ordinals)
    space._group_weights = weights
    space.dirty.clear()
    register_layouts(space, groups.keys(), checkpoint)
