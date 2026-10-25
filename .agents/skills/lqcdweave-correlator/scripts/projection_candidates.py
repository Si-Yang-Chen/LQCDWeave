"""Enumerate nonzero public little-group projections, without reading data.
Input JSON: {"groups": [{"label": str, "particles": [...],
"targets": [{"irrep": str, "row": int, "parity": int|null}],
"physics": {...}}]}. Quantum-number labels are annotations, not validated here.
"""
import argparse
import json
from pathlib import Path

from lqcdweave.operators.correlator.api import (
    HadronIrrep, Tag, hadron_little_group_projection,
)


def enumerate_candidates(payload):
    output = {"candidates": [], "empty_projections": [],
              "physics_labels_validated": False}
    for group in payload["groups"]:
        particles = [HadronIrrep(
            item["name"], item["momentum"], item["irrep"], item.get("parity"),
            Tag(i, item.get("time", 0)),
        ) for i, item in enumerate(group["particles"])]
        total = [sum(p.momentum[k] for p in particles) for k in range(3)]
        for target in group["targets"]:
            rows = hadron_little_group_projection(
                particles, target["irrep"], target["row"],
                parity=target.get("parity"), single_result=False,
            )
            record = {"group": group["label"], "total_momentum": total,
                      "particles": group["particles"], "target": target,
                      "physics": group.get("physics", {})}
            if not rows:
                output["empty_projections"].append(record)
            for index, expression in enumerate(rows):
                output["candidates"].append(dict(
                    record, id=f"candidate-{len(output['candidates']) + 1}",
                    basis_index=index, expression=str(expression),
                ))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = enumerate_candidates(payload)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
