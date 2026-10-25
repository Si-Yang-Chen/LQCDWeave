"""Operation-level measurement views for existing scene replay reports."""

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from ..contracts.measurement import ActionTiming, FileIOEvent, PhaseTiming, ReplayReport


def summarize_measurement(report):
    """Aggregate recorded atomic-operation timings without changing their scope."""
    totals = defaultdict(lambda: {"host_seconds": 0.0, "operations": 0})
    for timing in (*report.timings, *report.output_timings):
        row = totals[timing.kind]
        row["host_seconds"] += float(timing.observed_seconds)
        row["operations"] += 1
    return {name: dict(row) for name, row in totals.items()}


def save_measurement(report, path):
    """Persist the measured operation records for a later explicit fit."""
    if not isinstance(report, ReplayReport):
        raise TypeError("report must be a ReplayReport")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload.pop("values", None)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return target


def load_measurement(path):
    """Load an operation-level report saved by :func:`save_measurement`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    def action_timing(item):
        return ActionTiming(**{
            **item,
            "phases": tuple(PhaseTiming(**phase) for phase in item.get("phases", ())),
            "operands": tuple(item.get("operands", ())),
        })

    timings = tuple(action_timing(item) for item in payload.get("timings", ()))
    outputs = tuple(action_timing(item) for item in payload.get("output_timings", ()))
    events = tuple(FileIOEvent(**item) for item in payload.get("io_events", ()))
    return ReplayReport(tuple(payload.get("values", ())), timings,
                        payload.get("peak_device_bytes"), payload.get("peak_host_rss_bytes"),
                        outputs, events, payload.get("execution_profile"))


__all__ = ["load_measurement", "save_measurement", "summarize_measurement"]
