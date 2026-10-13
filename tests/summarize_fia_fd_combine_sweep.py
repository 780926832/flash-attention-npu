"""Summarize paired FIA FD combine tail differences and plot input sweeps.

Each point is mean(full Task Duration) - mean(no-combine Task Duration),
averaged across three paired rounds. Error bars show round min/max, not CIs.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root, allow_incomplete=False):
    cases = json.loads((root / "cases.json").read_text())
    runs = json.loads((root / "runs.json").read_text())
    schedules = json.loads((root / "schedules-full.json").read_text())["records"]
    assert len({r["device"] for r in runs}) == 1
    plans = {(r["case"], r["return_lse"]): r for r in schedules}
    indexed = {}
    for run in runs:
        for group in run["groups"]:
            key = group["case"], group["return_lse"], run["round"], run["variant"]
            assert key not in indexed, key
            indexed[key] = group
    results, pending = [], []
    for case in cases:
        if any((case["name"], lse, rid, variant) not in indexed
               for lse in (True, False) for rid in (1, 2, 3) for variant in ("full", "no-combine")):
            pending.append(case["name"])
            continue
        for lse in (True, False):
            pairs = []
            for rid in (1, 2, 3):
                full = indexed[case["name"], lse, rid, "full"]
                off = indexed[case["name"], lse, rid, "no-combine"]
                pairs.append(dict(round=rid, full_us=full["mean_us"], no_combine_us=off["mean_us"],
                                  delta_us=full["mean_us"] - off["mean_us"],
                                  median_delta_us=full["median_us"] - off["median_us"]))
            values = [p["delta_us"] for p in pairs]
            results.append(dict(case=case["name"], return_lse=lse, device=runs[0]["device"], source=root.name,
                                combine_us=statistics.mean(values),
                                round_min_us=min(values), round_max_us=max(values),
                                round_spread_us=max(values) - min(values),
                                full_us=statistics.mean(p["full_us"] for p in pairs),
                                no_combine_us=statistics.mean(p["no_combine_us"] for p in pairs),
                                rounds=pairs, metrics=plans[case["name"], lse]["metrics"]))
    if pending and not allow_incomplete:
        raise RuntimeError(f"Incomplete paired rounds: {pending}")
    write_summary(root, results)
    (root / "pending-timings.json").write_text(json.dumps(pending, indent=2) + "\n")
    return results, pending


def write_summary(root, results):
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    with (root / "summary.csv").open("w") as file:
        fields = ("case", "return_lse", "device", "source", "combine_us", "round_min_us", "round_max_us", "full_us", "no_combine_us",
                  "split_set", "row_tile_set", "active_aiv", "launched_aiv", "max_row_rounds",
                  "max_o_dma_calls", "max_o_read_KiB", "total_o_read_KiB", "total_o_dma_calls")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            merged = dict(item, **item["metrics"])
            writer.writerow({key: merged[key] for key in fields})


def summarize_series(root, manifest):
    """Keep each parameter sweep on one device, including its shared anchors."""
    sources = json.loads(manifest.read_text())
    sweeps = json.loads((root / "sweeps.json").read_text())
    assert sources.keys() == sweeps.keys()
    fd_cases = {c["name"] for c in json.loads((root / "cases.json").read_text())}
    combined, devices, loaded, selections = [], {}, {}, {}
    for family, relative in sources.items():
        source = (root / relative).resolve()
        if source not in loaded:
            records, pending = summarize(source)
            assert not pending
            loaded[source] = records
            combined.extend(records)
        records = loaded[source]
        available = {(r["case"], r["return_lse"]) for r in records}
        assert all((e["case"], lse) in available for e in sweeps[family]
                   if e["case"] in fd_cases for lse in (True, False)), family
        cards = {r["device"] for r in records}
        assert len(cards) == 1
        devices[family] = cards.pop()
        selections[family] = source.name
    assert len({(r["case"], r["return_lse"], r["source"]) for r in combined}) == len(combined)
    assert {r["case"] for r in combined} == fd_cases
    write_summary(root, combined)
    (root / "series-devices.json").write_text(json.dumps(devices, indent=2) + "\n")
    (root / "series-sources.json").write_text(json.dumps(selections, indent=2) + "\n")
    (root / "pending-timings.json").write_text("[]\n")
    return combined, []


def figure(root, results, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = json.loads((root / "sweeps.json").read_text())
    lookup = {(r["case"], r["return_lse"], r["source"]): r for r in results}
    devices_file = root / "series-devices.json"
    devices = json.loads(devices_file.read_text()) if devices_file.exists() else dict.fromkeys(sweeps, results[0]["device"])
    sources_file = root / "series-sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else dict.fromkeys(sweeps, results[0]["source"])
    labels = dict(q="Mixed KV: Q length (B4 / H8)", heads="Head count (B1 / Q1 / KV8192)",
                  kv="KV length (B1 / Q1 / H64)", batch="Batch count (Q4 / H8 / KV8192)",
                  skew="Auxiliary KV length (B4 / Q14 / H8)",
                  kvheads="KV head count (B1 / Q1 / H64 / KV8192)",
                  q_single="Q length (B1 / H8 / KV8192)", varq="Variable Q/KV patterns")
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    for ax, (family, entries) in zip(axes.flat, sweeps.items()):
        device = devices[family]
        source = sources[family]
        present = [e for e in entries if (e["case"], True, source) in lookup]
        for lse, color, marker in ((True, "#2266bb", "o"), (False, "#dd7733", "s")):
            selected = [lookup[e["case"], lse, source] for e in present]
            ax.errorbar(range(len(present)), [r["combine_us"] for r in selected],
                        yerr=[[r["combine_us"] - r["round_min_us"] for r in selected],
                              [r["round_max_us"] - r["combine_us"] for r in selected]],
                        color=color, marker=marker, markersize=4, capsize=3, linewidth=1.4,
                        label=f"LSE {'on' if lse else 'off'}")
        ax.set_xticks(range(len(present)), [str(e["value"]) for e in present],
                      rotation=40 if family in ("kv", "varq") else 0)
        ax.set_title(labels[family] + f" — NPU {device}", fontsize=11)
        ax.set_ylabel("Combine estimate (us)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
    fig.suptitle("FIA FD combine: paired full - disabled kernel durations\n"
                 "BF16 / D128 / TND / page128; points: 3-round mean; bars: min-max; FD cases only", fontsize=14)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--series-manifest", type=Path,
                        help="Map each sweep to an artifact directory; never mix cards within a sweep")
    args = parser.parse_args()
    if args.series_manifest:
        if args.allow_incomplete:
            parser.error("Series reports require complete paired rounds")
        results, pending = summarize_series(args.artifacts, args.series_manifest)
    else:
        results, pending = summarize(args.artifacts, args.allow_incomplete)
    if args.figure:
        figure(args.artifacts, results, args.figure)
    print(json.dumps(dict(complete_cases=len({r['case'] for r in results}),
                          measurement_groups=len(results), pending=len(pending))))


if __name__ == "__main__":
    main()
