"""Summarize paired msprof full/no-combine rounds and plot FD scaling curves."""

import argparse
import csv
import json
from pathlib import Path
import statistics as stats


def summarize(root, rounds, allow_incomplete=False):
    cases = json.loads((root / "cases.json").read_text())
    plans = {p["case"]: p for p in json.loads((root / "schedules.json").read_text())}
    runs = json.loads((root / "runs.json").read_text())
    lookup = {}
    for run in runs:
        for group in run["groups"]:
            key = (run["round"], run["variant"], group["case"], int(group["return_lse"]))
            assert key not in lookup
            lookup[key] = group
    output, pending = [], []
    for case in cases:
        missing = [(i, variant, lse) for i in range(1, rounds + 1)
                   for variant in ("full", "no-combine") for lse in (1, 0)
                   if (i, variant, case["name"], lse) not in lookup]
        if missing:
            if not allow_incomplete:
                raise ValueError(f"Incomplete paired rounds for {case['name']}: {missing}")
            pending.append(dict(case=case, missing=missing))
            continue
        for lse in (1, 0):
            means = {variant: [lookup[(i, variant, case["name"], lse)]["mean_us"]
                               for i in range(1, rounds + 1)]
                     for variant in ("full", "no-combine")}
            differences = [a - b for a, b in zip(means["full"], means["no-combine"])]
            plan = plans[case["name"]]
            metrics = {k: v for k, v in plan["metrics"][str(lse)].items() if k != "cores"}
            output.append(dict(case=case, return_lse=lse, **metrics,
                               partials=[s["partials"] for s in plan["host"]["schedules"]],
                               combine_rows=plan["host"]["combine_rows"],
                               full_us=stats.mean(means["full"]),
                               no_combine_us=stats.mean(means["no-combine"]),
                               combine_us=stats.mean(differences),
                               combine_rounds_us=differences,
                               combine_round_min_us=min(differences),
                               combine_round_max_us=max(differences),
                               means_by_round=means))
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "pending-timings.json").write_text(json.dumps(pending, indent=2) + "\n")
    with (root / "summary.csv").open("w") as f:
        fields = ["case", "family", "value", "dtype", "return_lse", "partials",
                  "combine_rows", "mode", "active", "launched", "total_KiB", "max_KiB",
                  "max_rounds", "max_dma_batches", "max_dma_per_round", "max_o_dma_calls",
                  "total_o_dma_calls", "row_extents",
                  "full_us", "no_combine_us", "combine_us", "combine_round_min_us",
                  "combine_round_max_us"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in output:
            value = {k: row[k] for k in fields if k in row and k != "case"}
            value.update(case=row["case"]["name"], family=row["case"]["family"],
                         value=row["case"]["value"], dtype=row["case"]["dtype"])
            writer.writerow(value)
    return output


def plot(rows, output, source_head):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [("q", "Query length / batch"), ("d", "Head dimension D"),
                ("heads", "Query heads, B=Q=KVH=1"),
                ("split_single", "Requested splits, single batch"),
                ("split_mixed", "Requested splits, mixed KV"),
                ("kv", "KV length, B=Q=KVH=1"),
                ("batch", "Uniform batch count"), ("skew", "Auxiliary KV length"),
                ("kvheads", "KV heads, H=64")]
    families = [(family, label) for family, label in families
                if len({r["case"]["value"] for r in rows if r["case"]["family"] == family}) > 1]
    assert families, "No completed sweep to plot"
    nrows = (len(families) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.6 * nrows + 1),
                             squeeze=False, constrained_layout=True)
    for ax, (family, label) in zip(axes.flat, families):
        for lse, color in ((1, "#1c6eaf"), (0, "#db7622")):
            data = sorted((r for r in rows if r["case"]["family"] == family
                           and r["return_lse"] == lse), key=lambda r: r["case"]["value"])
            values = [r["case"]["value"] for r in data]
            positions = list(range(len(values))) if family == "d" else values
            ax.errorbar(positions,
                        [r["combine_us"] for r in data],
                        yerr=[[r["combine_us"] - r["combine_round_min_us"] for r in data],
                              [r["combine_round_max_us"] - r["combine_us"] for r in data]],
                        marker="o", ms=4, capsize=2, color=color, label=f"LSE {['off', 'on'][lse]}")
            if family == "d":
                ax.set_xticks(positions, values, rotation=45, fontsize=8)
        if family in ("kv", "skew"):
            ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Estimated combine time (us)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.suptitle(f"Ascend950 FD combine, code {source_head}\n"
                 "Paired full - no-combine; error bars show min/max of round differences",
                 fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Report only complete paired rounds; write missing cases separately.")
    args = parser.parse_args()
    rows = summarize(args.artifacts, args.rounds, args.allow_incomplete)
    if args.figure:
        source_head = json.loads((args.artifacts / "provenance.json").read_text())["head"][:7]
        plot(rows, args.figure, source_head)
    print(f"Summarized {len(rows)} groups, {args.rounds} paired rounds each")


if __name__ == "__main__":
    main()
