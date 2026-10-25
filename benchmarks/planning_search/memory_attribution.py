"""Retained-memory attribution for a frozen planning archive (in-process only).

Loads one frozen storage archive, runs the integer fingerprint graph DFS at
bounded state budgets, and partitions the retained Python object graph into
owned categories with a greedy, order-documented walk. Two budgets are diffed
to report the marginal bytes/state of every category, and the registry is split
into applied and never-applied candidate recipes.

This is a diagnostic tool. It never runs benchmarks/planning_search/run.py,
never schedules storage and never measures contraction time. The archive stays
outside the repository.

The greedy order is part of the result: earlier categories own every object
they reach first, so growing per-state tables come first and the shared fixed
base (problem, physical, space constants) comes last. Both budgets use the same
order, so the diff isolates marginal growth.

    python benchmarks/planning_search/memory_attribution.py \
      --archive <frozen.storage.json.gz> --source . --states 1536 3072 --json out.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


def load_archive(archive):
    from lqcdweave.storage.contraction_plans.api import decode_plan_value
    payload = json.load(gzip.open(archive, "rt"))["payload"]
    return replace(decode_plan_value(payload["plans"][0]),
                   blocks=decode_plan_value(payload["blocks"]))


def process_memory():
    """Resident and peak resident bytes, or None when unavailable."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return dict(rss=info.rss, peak_wset=getattr(info, "peak_wset", None))
    except Exception:
        return None


def projection_overhead(rows):
    return sys.getsizeof(rows) + sum(sys.getsizeof(row) for row in rows)


def category_roots(sessions, registry, physical, applied):
    """Ordered category roots for the greedy retained-object partition."""
    spaces = [session.space for session in sessions]
    workspace, terminal, recipes = [], [], []
    for space in spaces:
        terminal.extend((
            space.terminal_families, space.terminal_members, space.terminal_heap,
            space.layout_variants, getattr(space, "_terminal_completions", None),
            getattr(space, "_scalar_terminals", None)))
        workspace.extend((
            space.rows, space.groups, space._group_weights, space.group_ordinals,
            space.dirty, space.available, space.catalog, space._ranking,
            getattr(space, "_prediction_catalog", None), space._dense_contexts))
        recipes.extend((space._dense_recipes, space._dense_predictions))
    calculations = registry.calculations
    # Projection: the same referenced objects without the recipe edges, so the
    # recipe closure can be attributed separately and honestly.
    records = [(entry.result, entry.seconds, entry.workspace) for entry in calculations]
    return (
        ("nodes", tuple(session.nodes for session in sessions)),
        ("connections", tuple(session.connections for session in sessions)),
        ("operations.pool", (registry.operations,)),
        ("operations.costs", (registry.operation_costs,
                              tuple(session.operation_costs for session in sessions))),
        ("registry.data", (registry.data, registry.data_names, registry.data_shape_keys)),
        ("registry.calculations", (records,)),
        ("registry.recipes", tuple(entry.recipe for entry in calculations)),
        ("registry.names", (registry.names, registry.core_ids)),
        ("registry.calculation_ids", (registry.calculation_ids,)),
        ("registry.patterns", (registry.patterns, registry.pattern_ids)),
        ("registry.indexes", (registry.layout_bases, registry.dense_pair_masks,
                              registry.dense_choices, registry.dense_hot,
                              registry.dense_relation_ids, registry.term_transitions)),
        ("space.workspace", tuple(workspace)),
        ("space.terminal", tuple(terminal)),
        ("space.recipe_cache", tuple(recipes)),
        ("space.terms", tuple((space.terms, space.term_hashes) for space in spaces)),
        ("problem", (physical.problem,)),
        ("physical", (physical,)),
        ("space.constants", tuple(space.memory_constants() for space in spaces)),
        ("frames", tuple(session.frames for session in sessions)),
        ("session.misc", tuple(sessions)),
    ), projection_overhead(records)


def container_breakdown(sessions, registry):
    """Exact container/record bytes for the growing per-state tables.

    Direct summation over the live containers avoids the greedy walk entirely,
    so the composition of the largest category does not depend on walk order.
    Shared identity bytes appear once, in the nodes table that owns them.
    """
    nodes = [session.nodes for session in sessions]
    connections = [session.connections for session in sessions]
    return dict(
        nodes_dict=sum(sys.getsizeof(table) for table in nodes),
        nodes_keys=sum(sys.getsizeof(key) for table in nodes for key in table),
        nodes_records=sum(sys.getsizeof(node) for table in nodes for node in table.values()),
        nodes_children=sum(sys.getsizeof(node.children) for table in nodes for node in table.values()),
        nodes_deferred=sum(sys.getsizeof(node.deferred) for table in nodes for node in table.values()),
        nodes_goals=sum(1 for table in nodes for node in table.values() if node.goal),
        connections_dict=sum(sys.getsizeof(table) for table in connections),
        connections_keys=sum(sys.getsizeof(key) for table in connections for key in table),
        connections_values=sum(sys.getsizeof(value) for table in connections for value in table.values()),
        operations_pool_dict=sys.getsizeof(registry.operations._ids),
        operations_pool_list=sys.getsizeof(registry.operations._operations))


def build_report(sessions, registry, physical, applied):
    from lqcdweave.contraction.planning.optimizers.budget import retained_walk
    roots, overhead = category_roots(sessions, registry, physical, applied)
    seen, report, total = set(), {}, 0
    for name, category in roots:
        before = total
        total, objects = retained_walk(category, seen, total)
        if name == "registry.calculations":
            total -= overhead
            objects -= len(registry.calculations)
        report[name] = dict(bytes=total - before, objects=objects)
    return report, total


def recipe_split(registry, applied):
    """Applied versus never-applied candidate records and recipe closures."""
    from lqcdweave.contraction.planning.optimizers.budget import retained_walk
    calculations = registry.calculations
    split = {}
    for name, keep in (("all", lambda index: True), ("applied", applied.__contains__),
                       ("unapplied", lambda index: index not in applied)):
        entries = [entry for index, entry in enumerate(calculations) if keep(index)]
        records = [(entry.result, entry.seconds, entry.workspace) for entry in entries]
        recipes = [entry.recipe for entry in entries]
        seen, total = set(), 0
        total, objects = retained_walk(records, seen, total)
        split[name] = dict(count=len(entries), record_bytes=total - projection_overhead(records),
                           record_objects=objects - len(records))
        total, objects = retained_walk(recipes, seen, total)
        split[name]["recipe_bytes"] = total - split[name]["record_bytes"] - projection_overhead(records)
        split[name]["recipe_objects"] = objects - split[name]["record_objects"] - len(records)
    return split


def run_budget(baseline, states, *, source, deadline, sampling, scorer, memory_limit, workers):
    from lqcdweave.contraction.planning.optimizers import graph_session
    from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
    from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace

    problem = baseline.state.problem
    captured = {"sessions": [], "applied": set()}

    class _Stop(Exception):
        pass

    original_advance = FingerprintDFS.advance
    original_apply = IntegerSpace.apply

    class RecordingDFS(FingerprintDFS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["sessions"].append(self)

        def advance(self, *, first_solution=False):
            return original_advance(self, first_solution=first_solution)

    def apply(space, operation, checkpoint):
        captured["applied"].add(operation.calculation)
        return original_apply(space, operation, checkpoint)

    graph_session.FingerprintDFS = RecordingDFS
    IntegerSpace.apply = apply

    def finish(physical, audited, snippets, workspaces, **kwargs):
        sessions = captured["sessions"]
        registry = sessions[0].space.registry
        space = sessions[0].space
        # Measure the process before attribution allocates its own projections.
        captured["memory"] = process_memory()
        started = time.perf_counter()
        report, total = build_report(sessions, registry, physical, captured["applied"])
        captured["attribution_seconds"] = time.perf_counter() - started
        captured["report"] = report
        captured["total"] = total
        captured["objects"] = sum(row["objects"] for row in report.values())
        captured["recipes"] = recipe_split(registry, captured["applied"])
        captured["containers"] = container_breakdown(sessions, registry)
        captured["sampling"] = sessions[0].budget.memory_sampling_statistics
        captured["counters"] = dict(
            nodes=sum(len(session.nodes) for session in sessions),
            connections=sum(len(session.connections) for session in sessions),
            applied_calls=space.applied, enumerations=space.enumerations,
            pooled_operations=len(registry.operations),
            data=len(registry.data), calculations=len(registry.calculations),
            patterns=len(registry.patterns), names=len(registry.names),
            dense_choices=len(registry.dense_choices), dense_hot=len(registry.dense_hot),
            term_transitions=len(registry.term_transitions),
            dense_recipes=len(space._dense_recipes))
        raise _Stop()

    graph_session.finish_fixed_storage = finish
    started = time.perf_counter()
    try:
        graph_session.solve_graph_storage(
            problem, baseline.blocks, baseline.preparations, profile=baseline.cost_profile,
            memory_limit=baseline.memory_limit, host_memory_limit=baseline.host_memory_limit,
            component_workers=workers, root_width=4, branch_width=4, include_ties=True,
            scorer=scorer, deadline_seconds=deadline, max_states=states,
            search_memory_limit=memory_limit, memory_sampling=sampling, kernel_batching=False)
    except _Stop:
        pass
    captured["wall_seconds"] = time.perf_counter() - started
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--states", type=int, nargs="+", default=[1536, 3072])
    parser.add_argument("--deadline", type=float, default=3600.0)
    parser.add_argument("--sampling", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--scorer", default="time_per_term")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-limit", type=int, default=1 << 40)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    started = time.perf_counter()
    baseline = load_archive(args.archive)
    load_seconds = time.perf_counter() - started
    after_load = process_memory()
    runs = []
    for states in args.states:
        captured = run_budget(baseline, states, source=args.source, deadline=args.deadline,
                              sampling=args.sampling, scorer=args.scorer,
                              memory_limit=args.memory_limit, workers=args.workers)
        run = dict(states=states, expanded=captured["counters"]["nodes"],
                   wall_seconds=round(captured["wall_seconds"], 3),
                   attribution_seconds=round(captured["attribution_seconds"], 3),
                   total_bytes=captured["total"], total_objects=captured["objects"],
                   categories=captured["report"], recipes=captured["recipes"],
                   containers=captured["containers"], sampling=captured["sampling"],
                   counters=captured["counters"], memory=captured["memory"])
        runs.append(run)
        schedule = run["sampling"] or {}
        print(f"states={run['expanded']} total={run['total_bytes']:,}B "
              f"objects={run['total_objects']:,} wall={run['wall_seconds']}s "
              f"peak_wset={run['memory'] and run['memory']['peak_wset']} "
              f"walks={schedule.get('walks')} "
              f"walk_seconds={schedule.get('seconds') and round(schedule['seconds'], 3)} "
              f"walk_objects={schedule.get('visited_objects')}", flush=True)
    document = dict(archive=str(args.archive), sampling=args.sampling,
                    load_seconds=round(load_seconds, 3), memory_after_load=after_load, runs=runs)
    for index in range(1, len(runs)):
        low, high = runs[index - 1], runs[index]
        # Retained states, not requested budgets, are the growth denominator.
        delta_states = high["expanded"] - low["expanded"]
        table = {}
        for name in low["categories"]:
            a, b = low["categories"][name], high["categories"][name]
            table[name] = dict(low_bytes=a["bytes"], high_bytes=b["bytes"],
                               bytes_per_state=round((b["bytes"] - a["bytes"]) / delta_states, 3),
                               objects_per_state=round((b["objects"] - a["objects"]) / delta_states, 3))
        for name in low["containers"]:
            a, b = low["containers"][name], high["containers"][name]
            table["container." + name] = dict(low_bytes=a, high_bytes=b,
                bytes_per_state=round((b - a) / delta_states, 3), objects_per_state=0.0)
        # Sampling cost is reported in seconds and visited objects per retained
        # state; both are derived from the same two budgets as the byte table.
        if low["sampling"] and high["sampling"]:
            table["sampling.walk_seconds"] = dict(
                low_bytes=low["sampling"]["seconds"], high_bytes=high["sampling"]["seconds"],
                bytes_per_state=round((high["sampling"]["seconds"] - low["sampling"]["seconds"])
                                      / delta_states * 1e6, 3),
                objects_per_state=round((high["sampling"]["visited_objects"]
                                         - low["sampling"]["visited_objects"]) / delta_states, 3))
        document[f"delta_{low['expanded']}_{high['expanded']}"] = table
        print(f"delta {low['expanded']}->{high['expanded']} retained states "
              f"({delta_states}) total {round((high['total_bytes'] - low['total_bytes']) / delta_states, 1)} B/state:")
        for name, row in table.items():
            unit = "us/state" if name == "sampling.walk_seconds" else "B/state "
            scale = "s" if name == "sampling.walk_seconds" else ""
            print(f"  {name:36s} low={row['low_bytes']:>16,.3f}{scale} high={row['high_bytes']:>16,.3f}{scale} "
                  f"{row['bytes_per_state']:>9.3f} {unit} {row['objects_per_state']:>7.3f} obj/state")
    if args.json is not None:
        args.json.write_text(json.dumps(document, indent=2))
    return document


if __name__ == "__main__":
    main()
