"""Summarize complete paired FA measurements with a shared no-combine control."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--candidate", default="overlap")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--control", default="baseline-no-combine")
    args = parser.parse_args()
    root = args.artifacts.resolve()
    config = json.loads((root / "collection-config.json").read_text())
    runs = json.loads((root / "runs.json").read_text())
    by = {(r["round"], r["variant"]): r for r in runs}
    assert len(by) == len(runs), "duplicate captures"
    variants = (args.baseline, args.candidate, args.control)
    rounds = range(1, config["rounds"] + 1)
    reference = by[1, args.baseline]["metadata"]
    for rid in rounds:
        for variant in variants:
            record = by[rid, variant]
            assert record["device"] == config["device"]
            assert record["metadata"]["input_sha256"] == reference["input_sha256"]
            assert record["metadata"]["groups"] == reference["groups"]
    output = []
    for idx, group in enumerate(reference["groups"]):
        values = {v: [by[r, v]["groups"][idx]["mean_us"] for r in rounds] for v in variants}
        baseline, candidate, control = (values[v] for v in variants)
        gain = [a - b for a, b in zip(baseline, candidate)]
        bc = [a - b for a, b in zip(baseline, control)]
        cc = [a - b for a, b in zip(candidate, control)]
        mean_base, mean_candidate = statistics.mean(baseline), statistics.mean(candidate)
        output.append(dict(group, baseline_full_us=mean_base, candidate_full_us=mean_candidate,
                           baseline_full_rounds_us=baseline, candidate_full_rounds_us=candidate,
                           control_rounds_us=control, full_gain_us=statistics.mean(gain),
                           full_gain_rounds_us=gain,
                           full_gain_pct=100 * statistics.mean(gain) / mean_base,
                           baseline_combine_us=statistics.mean(bc), candidate_combine_us=statistics.mean(cc),
                           baseline_combine_rounds_us=bc, candidate_combine_rounds_us=cc,
                           combine_gain_pct=100 * statistics.mean(gain) / statistics.mean(bc)))
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["| case | LSE | full 基线 µs | full 候选 µs | full 提升 | combine 基线 µs | combine 候选 µs | 三轮 full 节省 µs |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in output:
        gains = " / ".join(f"{x:+.3f}" for x in row["full_gain_rounds_us"])
        lines.append(f"| {row['case']} | {int(row['return_lse'])} | {row['baseline_full_us']:.3f} | "
                     f"{row['candidate_full_us']:.3f} | {row['full_gain_pct']:+.2f}% | "
                     f"{row['baseline_combine_us']:.3f} | {row['candidate_combine_us']:.3f} | {gains} |")
    (root / "summary-table.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(output)} groups across {config['rounds']} rounds on NPU {config['device']}")


if __name__ == "__main__":
    main()
