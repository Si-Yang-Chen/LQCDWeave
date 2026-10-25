"""Strict mode-specific search options; exact and DFS domains stay separate."""
from .budget import DEFAULT_PLANNING_SECONDS, integer
from .integer_scoring import normalize_scorer


def resolve_strategy_options(optimizer="anytime", **options):
    if optimizer not in ("anytime", "exact", "dynamic_programming"):
        raise ValueError("optimizer must be anytime, exact or dynamic_programming")
    defaults = dict(root_width=4, branch_width=4, include_ties=True, disk_limit_bytes=None, debug=False,
                    scorer="shared_savings", profile_scoring=False, component_workers=None,
                    storage_branch_width=4)
    allowed = defaults.keys() if optimizer == "anytime" else ()
    unknown = options.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unsupported options for {optimizer}: {', '.join(sorted(unknown))}")
    if optimizer != "anytime":
        return {}
    defaults.update(options)
    integer(defaults["root_width"], "root_width", 1)
    integer(defaults["branch_width"], "branch_width", 1)
    integer(defaults["storage_branch_width"], "storage_branch_width", 1)
    if defaults["component_workers"] is not None:
        integer(defaults["component_workers"], "component_workers", 1)
    if type(defaults["include_ties"]) is not bool:
        raise ValueError("include_ties must be bool")
    if type(defaults["debug"]) is not bool:
        raise ValueError("debug must be bool")
    defaults["scorer"] = normalize_scorer(defaults["scorer"])
    if type(defaults["profile_scoring"]) is not bool:
        raise ValueError("profile_scoring must be bool")
    if defaults["disk_limit_bytes"] is not None:
        integer(defaults["disk_limit_bytes"], "disk_limit_bytes", 1)
    return defaults


SEARCH_CONTROL_NAMES = frozenset(("deadline_seconds", "max_states", "max_transitions",
                                "search_memory_limit", "memory_sampling", "clock"))

# "fixed" walks the complete retained object graph on the former fixed cadence.
# It is the debug comparison for the adaptive schedule and is much slower.
MEMORY_SAMPLING_MODES = ("adaptive", "fixed")


def effective_search_options(optimizer, supplied):
    strategies = {key: value for key, value in supplied.items() if key not in SEARCH_CONTROL_NAMES}
    effective = dict(deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000, max_transitions=None,
                     search_memory_limit=None, memory_sampling="adaptive")
    effective.update({key: value for key, value in supplied.items() if key in SEARCH_CONTROL_NAMES})
    if effective["memory_sampling"] not in MEMORY_SAMPLING_MODES:
        raise ValueError("memory_sampling must be 'adaptive' or 'fixed'")
    effective.update(resolve_strategy_options(optimizer, **strategies))
    return effective
