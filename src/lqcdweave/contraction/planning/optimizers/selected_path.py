"""Materialize selected dense recipes without enumerating discarded choices.

The registry owns the descriptions accepted during search. An operand index
finds their current occurrences; axis matching binds those immutable recipes
to the original Terms. A route whose operations already carry search-time
occurrence bindings skips that axis matching and its intermediate expression
objects; independent costing and expression replay remain in IntegerSpace's
debug path and in this module's recompute/verify options.
"""
from collections import defaultdict, OrderedDict
from dataclasses import replace
from itertools import combinations
import math
import numpy as np

from lqcdweave.contracts.correlator.api import Action, AxisPattern, Factor, PendingTerm, FactorizedWideReduction
from ..canonical_operations import axis_key, output_axes
from ..state import ContractionState, _nbytes
from ..sum_boundaries import closed_sum_mode


def _occurrence(term, recipe, key, *, barriers=(), catalog=None):
    names = frozenset(recipe.operands)
    positions = [i for i, factor in enumerate(term.factors) if factor.tensor in names]
    choices = ((0,),) if len(term.factors) == 1 else combinations(positions, len(recipe.operands))
    for pair in choices:
        selected = tuple(term.factors[i] for i in pair)
        inputs = tuple(f.tensor for f in selected)
        if sorted(inputs) != sorted(recipe.operands):
            continue
        if type(recipe.core) is FactorizedWideReduction and inputs != recipe.operands:
            continue
        labels = tuple(f.labels for f in selected)
        needed = set(term.output)
        for i, factor in enumerate(term.factors):
            if i not in pair:
                needed.update(factor.labels)
        retained = needed.intersection(axis for axes in labels for axis in axes)
        # Direct wide candidates close the whole Term in its declared output
        # order. Binary candidates instead choose their canonical intermediate
        # order. Re-deriving the latter for a terminal core can transpose its
        # axes merely because an operand name sorts differently.
        terminal_core = type(recipe.core) is FactorizedWideReduction and len(pair) == len(term.factors)
        output = term.output if len(pair) == 1 or terminal_core else output_axes(inputs, labels, retained)
        if axis_key(inputs, labels, output) == key:
            if catalog is not None and (barriers or recipe.wide_accumulation):
                wide = False
                dtype = np.result_type(*(catalog[name].dtype for name in inputs))
                if barriers and dtype in (np.dtype('float32'), np.dtype('complex64')):
                    before = {axis for factor in term.factors for axis in factor.labels}
                    after = set(output) | {axis for i, factor in enumerate(term.factors)
                                           if i not in pair for axis in factor.labels}
                    wide = closed_sum_mode(before, after, barriers)
                    if wide is None:
                        continue
                    if type(recipe.core) is FactorizedWideReduction:
                        # Re-derive reduced axes from the matched ordered roles;
                        # dummy labels need not equal the registered spelling.
                        needed = set(output).union(*(labels[i] for i in recipe.core.small_operands))
                        reduced = {a for i in recipe.core.large_operands for a in labels[i] if a not in needed}
                        triggers = set().union(*(set(trigger) for trigger, _ in barriers
                            if before.intersection(trigger) and not after.intersection(trigger)))
                        if not reduced <= triggers:
                            continue
                if wide != recipe.wide_accumulation:
                    continue
            return pair, output
    return None


def _matches(pending, scenes, calculation, registry, *, problem=None, catalog=None):
    recipe = registry.calculations[calculation].recipe
    key = axis_key(recipe.operands, recipe.equation.inputs, recipe.equation.output)
    matches = _match_rows(pending, scenes, recipe, key, problem=problem, catalog=catalog)
    # A currently canonical group takes precedence over optional output layouts,
    # exactly as register_layouts() does before adding an absent variant group.
    if matches:
        return matches
    for base, permutation in registry.layout_bases.get(calculation, ()):
        parent = registry.calculations[base].recipe
        parent_key = axis_key(parent.operands, parent.equation.inputs, parent.equation.output)
        members = _match_rows(pending, scenes, parent, parent_key, problem=problem, catalog=catalog)
        if not members or any(len(pending[scene].factors) != 2 or len(pair) != 2
                              for scene, (pair, _) in members):
            continue
        if any(tuple(output[i] for i in permutation) == pending[scene].output
               for scene, (_, output) in members):
            return [(scene, (pair, tuple(output[i] for i in permutation)))
                    for scene, (pair, output) in members]
    return []


def _match_rows(pending, scenes, recipe, key, *, problem=None, catalog=None):
    """Share exact axis/operand-role matches; unrelated names do not enter them."""
    names = frozenset(recipe.operands)
    memo, result = {}, []
    for scene in sorted(scenes):
        term = pending[scene]
        barriers = () if problem is None else problem.terms[scene].sum_barriers
        signature = (tuple((f.tensor if f.tensor in names else None, f.labels)
                           for f in term.factors), term.output, barriers)
        try:
            match = memo[signature]
        except KeyError:
            match = _occurrence(term, recipe, key, barriers=barriers, catalog=catalog)
            if len(memo) < 8192:
                memo[signature] = match
        if match is not None:
            result.append((scene, match))
    return result


def _dependent_scenes(consumers, recipe):
    """Scenes that still hold every operand; the exact-match pass filters axes."""
    dependencies = sorted((consumers[name] for name in set(recipe.operands)), key=len)
    return set.intersection(*dependencies)


def _materialize_bound_path(problem, operations, expected_cost, registry, bindings):
    """Audit a route whose every operation already carries its own bindings.

    The search recorded positions, operand names, axis relations and output
    order for each occurrence while its expression was live. Consuming them
    removes the independent axis derivation and the intermediate expression
    objects from the hot loop; completion, cost and peak accounting are the
    same as the full replay. The debug verify path rebuilds every Action through
    the ordinary frozen constructor and compares the two sequences field by
    field, so the skipped normalization stays checkable.
    """
    catalog = dict(problem.specs)
    loaded = set(catalog)
    # A header state prices the initial resident set without materializing the
    # per-Term expression objects that a fully bound route never reads.
    header = ContractionState(problem, catalog, (), frozenset(loaded), ())
    resident = header.resident_bytes
    outputs = tuple(term.subscripts.output for term in problem.terms)
    counts = [len(term.subscripts.inputs) for term in problem.terms]
    results = [None] * len(outputs)
    peak, seconds, actions, snippets = 0, 0.0, [], []
    equations = {}
    append = actions.append
    for ordinal, operation in enumerate(operations):
        calculation = registry.calculations[operation.calculation]
        definition = registry.data[operation.result]
        reused = definition.name in loaded
        if (operation.result != calculation.result or operation.reused != reused
                or operation.seconds != (0.0 if reused else calculation.seconds)):
            raise ValueError("integer operation changed its registered cost or availability")
        recipe_values = calculation.recipe.__dict__
        increment = calculation.seconds
        begin = len(actions)
        size = _nbytes(definition.spec)
        peak = max(peak, resident + (0 if reused else size + calculation.workspace))
        if not reused:
            catalog[definition.name] = definition.spec
            loaded.add(definition.name)
            resident += size
            seconds += calculation.seconds
        for scene, positions, output, operands, inputs in bindings[ordinal]:
            relation = inputs, output
            equation = equations.get(relation)
            if equation is None:
                equation = AxisPattern(*relation)
                if len(equations) >= 8192:
                    del equations[next(iter(equations))]
                equations[relation] = equation
            # Same object the frozen constructor would build: every substituted
            # value is already normalized, so no boundary pass is needed.
            action = Action.__new__(Action)
            values = action.__dict__
            values.update(recipe_values)
            values['tensor'] = definition.name
            values['operands'] = operands
            values['equation'] = equation
            if type(values.get('core')) is FactorizedWideReduction:
                values['core'] = replace(values['core'], equation=equation)
            values['term_index'] = scene
            values['positions'] = positions
            values['reused'] = reused
            values['seconds'] = 0.0 if reused else increment
            append(action)
            counts[scene] -= len(positions) - 1
            if counts[scene] == 1 and output == outputs[scene]:
                results[scene] = definition.name
            # Only the first occurrence of a newly computed result pays it.
            reused = True
        snippets.append((begin, len(actions)))
    if any(result is None for result in results) or not math.isclose(
            seconds, expected_cost, rel_tol=1e-11, abs_tol=1e-14):
        raise RuntimeError("selected integer path failed final completion or cost audit")
    state = ContractionState(problem, catalog, (None,) * len(results),
        frozenset(loaded), tuple(results), actions=tuple(actions), seconds=seconds,
        peak_bytes=peak, peak_host_bytes=peak if problem.backend == "numpy" else 0,
        peak_device_bytes=peak if problem.backend == "cupy" else 0)
    return state, snippets


def materialize_dense_path(problem, operations, expected_cost, registry, *, bindings=None,
                           recompute=False, verify=False):
    """Build Actions once from accepted recipes and freeze one final state.

    ``bindings`` optionally carries the search-time occurrence rows for this
    exact route, one entry per operation as ``(scene, positions, output,
    operands, inputs)`` with global scene IDs and ``None`` where no snapshot
    exists. Those rows are the axis relation the search already bound, so
    consuming them skips the independent operand/axis derivation. A route
    without a complete, usable snapshot sequence falls back to the independent
    match per operation. ``recompute=True`` ignores every snapshot and always
    derives it. ``verify=True`` re-derives it beside each snapshot and fails on
    the first disagreement; it is the debug entry for the skipped derivation.
    """
    if type(recompute) is not bool or type(verify) is not bool:
        raise ValueError("recompute and verify must be bool")
    if (bindings is not None and not recompute and not verify
            and len(bindings) == len(operations) and all(bindings)):
        # Every operation already carries the search's own occurrence rows, so
        # the whole route can skip axis matching and intermediate expressions.
        return _materialize_bound_path(problem, operations, expected_cost, registry, bindings)
    initial = ContractionState.initial(problem)
    pending, results = list(initial.pending), list(initial.results)
    consumers = defaultdict(set)
    for scene, term in enumerate(pending):
        for name in {factor.tensor for factor in term.factors}:
            consumers[name].add(scene)
    catalog = dict(initial.catalog)
    loaded = set(catalog)
    resident = replace(initial, loaded=frozenset(loaded)).resident_bytes
    peak, seconds, actions, snippets = 0, 0.0, [], []
    equations = OrderedDict()
    for ordinal, operation in enumerate(operations):
        calculation = registry.calculations[operation.calculation]
        definition = registry.data[operation.result]
        reused = definition.name in loaded
        if (operation.result != calculation.result or operation.reused != reused
                or operation.seconds != (0.0 if reused else calculation.seconds)):
            raise ValueError("integer operation changed its registered cost or availability")
        recipe = calculation.recipe
        begin = len(actions)
        binding = (None if recompute or bindings is None or ordinal >= len(bindings)
                   else bindings[ordinal])
        if binding:
            matches = sorted((scene, (positions, output))
                             for scene, positions, output, _, _ in binding)
            if verify:
                reference = _matches(pending, _dependent_scenes(consumers, recipe),
                                     operation.calculation, registry, problem=problem, catalog=catalog)
                if matches != reference:
                    raise RuntimeError("search-time occurrence bindings disagree with "
                                       "independent axis matching")
        else:
            # No usable snapshot: fall back to the complete independent match.
            matches = _matches(pending, _dependent_scenes(consumers, recipe),
                               operation.calculation, registry, problem=problem, catalog=catalog)
        if not matches:
            raise RuntimeError("selected hash path has no independently matching operation")
        size = _nbytes(definition.spec)
        peak = max(peak, resident + (0 if reused else size + calculation.workspace))
        result_factors = {}
        for scene, (positions, output) in matches:
            term = pending[scene]
            selected = tuple(term.factors[i] for i in positions)
            relation = tuple(f.labels for f in selected), output
            equation = equations.get(relation)
            if equation is None:
                equation = AxisPattern(*relation)
                equations[relation] = equation
                if len(equations) > 8192:
                    equations.popitem(last=False)
            else:
                equations.move_to_end(relation)
            action = replace(recipe, tensor=definition.name,
                operands=tuple(f.tensor for f in selected),
                equation=equation,
                term_index=scene, positions=positions, reused=reused,
                seconds=0.0 if reused else calculation.seconds)
            if type(action.core) is FactorizedWideReduction:
                action = replace(action, core=replace(action.core, equation=equation))
            actions.append(action)
            updated = tuple(f for i, f in enumerate(term.factors) if i not in positions)
            factor = result_factors.get(output)
            if factor is None:
                factor = result_factors[output] = Factor(definition.name, output)
            updated += (factor,)
            complete = len(updated) == 1 and output == term.output
            pending[scene] = None if complete else PendingTerm(updated, term.output)
            if complete:
                results[scene] = definition.name
            before_names = {f.tensor for f in term.factors}
            after_names = set() if complete else {f.tensor for f in updated}
            for name in before_names - after_names:
                consumers[name].remove(scene)
            for name in after_names - before_names:
                consumers[name].add(scene)
            if not reused:
                catalog[definition.name] = definition.spec
                loaded.add(definition.name)
                resident += size
                seconds += calculation.seconds
                reused = True
        snippets.append((begin, len(actions)))
    if any(result is None for result in results) or not math.isclose(
            seconds, expected_cost, rel_tol=1e-11, abs_tol=1e-14):
        raise RuntimeError("selected integer path failed final completion or cost audit")
    state = replace(initial, catalog=catalog, pending=tuple(pending), results=tuple(results),
        loaded=frozenset(loaded), actions=tuple(actions), seconds=seconds, peak_bytes=peak,
        peak_host_bytes=peak if problem.backend == "numpy" else 0,
        peak_device_bytes=peak if problem.backend == "cupy" else 0)
    return state, snippets
