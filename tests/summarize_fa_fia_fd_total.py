"""Summarize complete, same-device FA/FIA FD total-kernel measurements.

Each point is the mean of all measured calls, with equal calls in each round.
Round ranges describe reproducibility, not confidence intervals. FIA non-FD
routes are retained in a separate table and excluded from FD/FD aggregates.
"""

import argparse
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics


def all_kernel_samples(run):
    """Account for FA's per-call LSE Fill, excluding one-time input preparation.

    All calls synchronize; expected same-stream kernel sequences are checked
    exhaustively. No kernel after the first call may be silently discarded.
    """
    meta, variant = run["metadata"], run["variant"]
    with open(run["op_summary"]) as file:
        rows = sorted(csv.DictReader(file), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    attention = "FusedInferAttentionScore" if variant == "fia" else "FAInfer"
    first = next(i for i, r in enumerate(rows) if attention in r["Op Name"])
    start = first - int(variant != "fia" and meta["groups"][0]["return_lse"])
    prefix = rows[:start]
    assert all(any(k in r["Op Name"] for k in ("Arange", "Transpose")) for r in prefix)
    rows = rows[start:]
    pos, samples = 0, []
    for g in meta["groups"]:
        for call in range(meta["warmup"] + meta["measured"]):
            initialization = None
            if variant != "fia" and g["return_lse"]:
                initialization = rows[pos]
                assert "aclnnInplaceFillScalar_FillAiCore_Fill" == initialization["Op Name"]
                pos += 1
            main = rows[pos]
            assert attention in main["Op Name"]
            if initialization:
                assert initialization["Stream ID"] == main["Stream ID"]
            pos += 1
            if call < meta["warmup"]:
                continue
            init_us = float(initialization["Task Duration(us)"]) if initialization else 0.0
            main_us = float(main["Task Duration(us)"])
            samples.append(dict(round=run["round"], variant=variant, case=g["case"],
                                return_lse=g["return_lse"], sample=call - meta["warmup"],
                                main_us=main_us, initialization_us=init_us,
                                total_us=main_us + init_us, kernels=1 + int(initialization is not None)))
    assert pos == len(rows), "Unaccounted kernels inside or after attention calls"
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    cases = json.loads((root / "cases.json").read_text())
    config = json.loads((root / "collection-config.json").read_text())
    runs = json.loads((root / "runs.json").read_text())
    variants = ("fa0", "fa16", "fia")
    expected_runs = {(r, v) for r in range(1, config["rounds"] + 1) for v in variants}
    assert len(runs) == len(expected_runs)
    assert {(r["round"], r["variant"]) for r in runs} == expected_runs
    assert {r["device"] for r in runs} == {config["device"]}
    expected_groups = {(c["name"], lse) for c in cases for lse in (True, False)}
    by_group = {}
    all_samples = []
    reference = json.loads((root / "validation-fa0.json").read_text())["metadata"]
    for run in runs:
        samples = all_kernel_samples(run)
        all_samples.extend(samples)
        assert len(run["groups"]) == len(expected_groups)
        assert {(g["case"], g["return_lse"]) for g in run["groups"]} == expected_groups
        for name, digest in run["metadata"]["input_sha256"].items():
            assert digest == reference["input_sha256"][name]
        for g in run["groups"]:
            assert g["samples"] == config["iterations"]
            selected = [s for s in samples if (s["case"], s["return_lse"]) == (g["case"], g["return_lse"])]
            assert len(selected) == config["iterations"]
            assert abs(statistics.mean(s["main_us"] for s in selected) - g["mean_us"]) < 1e-8
            totals = [s["total_us"] for s in selected]
            assert max(totals) < 3 * statistics.median(totals), "Total-kernel timing tail needs review"
            by_group[run["round"], run["variant"], g["case"], g["return_lse"]] = dict(
                g, main_mean_us=g["mean_us"], mean_us=statistics.mean(totals),
                median_us=statistics.median(totals),
                initialization_mean_us=statistics.mean(s["initialization_us"] for s in selected))
    fia = {(r["case"], r["return_lse"]): r for r in
           json.loads((root / "schedules-fia.json").read_text())["records"]}
    fa = {v: {(r["case"], r["return_lse"]): r["host"] for r in
              json.loads((root / f"schedules-{v}.json").read_text())} for v in variants[:2]}
    results = []
    for c in cases:
        for lse in (True, False):
            key = c["name"], lse
            assert all(fa[v][key]["flash_decode"] for v in variants[:2])
            rec = dict(case=c["name"], return_lse=lse, family=c["family"],
                       fia_fd=fia[key]["is_fd"], shape=c,
                       fia_splits=fia[key].get("metrics", {}).get("split_set", []),
                       fa0_splits=sorted({s["partials"] for s in fa["fa0"][key]["schedules"]}),
                       fa16_splits=sorted({s["partials"] for s in fa["fa16"][key]["schedules"]}))
            for v in variants:
                groups = [by_group[r, v, *key] for r in range(1, config["rounds"] + 1)]
                means = [g["mean_us"] for g in groups]
                rec[v] = dict(mean_us=statistics.mean(means), round_means_us=means,
                              min_round_us=min(means), max_round_us=max(means),
                              round_medians_us=[g["median_us"] for g in groups],
                              samples=sum(g["samples"] for g in groups),
                              main_mean_us=statistics.mean(g["main_mean_us"] for g in groups),
                              initialization_mean_us=statistics.mean(g["initialization_mean_us"] for g in groups),
                              main_round_means_us=[g["main_mean_us"] for g in groups])
            rec["fa0_over_fia"] = rec["fa0"]["mean_us"] / rec["fia"]["mean_us"]
            rec["fa16_over_fia"] = rec["fa16"]["mean_us"] / rec["fia"]["mean_us"]
            rec["fa0_slower_pct"] = (rec["fa0_over_fia"] - 1) * 100
            rec["fa16_vs_fa0_pct"] = (rec["fa16"]["mean_us"] / rec["fa0"]["mean_us"] - 1) * 100
            results.append(rec)
    aggregates = []
    for fd in (True, False):
        for lse in (True, False):
            selected = [r for r in results if r["fia_fd"] == fd and r["return_lse"] == lse]
            if not selected:
                continue
            for v in variants[:2]:
                ratios = [r[v]["mean_us"] / r["fia"]["mean_us"] for r in selected]
                aggregates.append(dict(fia_fd=fd, return_lse=lse, variant=v, cases=len(selected),
                                       fa_faster=sum(x < 1 for x in ratios),
                                       ratio_geomean=math.exp(statistics.mean(map(math.log, ratios))),
                                       ratio_min=min(ratios), ratio_max=max(ratios),
                                       fa_mean_us=statistics.mean(r[v]["mean_us"] for r in selected),
                                       fia_mean_us=statistics.mean(r["fia"]["mean_us"] for r in selected)))
    summary = dict(config=config, cases=len(cases), valid_captures=len(runs),
                   measured_samples=sum(g["samples"] for r in runs for g in r["groups"]),
                   measured_kernels=sum(s["kernels"] for s in all_samples),
                   max_sample_over_median=max(g["max_us"] / g["median_us"] for r in runs for g in r["groups"]),
                   aggregates=aggregates, results=results)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "all-kernel-durations.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_samples[0]))
        writer.writeheader()
        writer.writerows(all_samples)
    flat = []
    for r in results:
        f = {k: r[k] for k in ("case", "return_lse", "family", "fia_fd", "fa0_over_fia", "fa16_over_fia")}
        f.update({k: ",".join(map(str, r[k])) for k in ("fa0_splits", "fa16_splits", "fia_splits")})
        for v in variants:
            f[v + "_mean_us"] = r[v]["mean_us"]
            f[v + "_main_mean_us"] = r[v]["main_mean_us"]
            f[v + "_initialization_mean_us"] = r[v]["initialization_mean_us"]
            for i, value in enumerate(r[v]["round_means_us"], 1):
                f[f"{v}_round{i}_us"] = value
        flat.append(f)
    with (root / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["## 输入矩阵", "", "Q/KV 列是各 batch 的实际长度。全部 page size=128，非 causal，无 mask。", "",
             "| case | dtype | B | Q | Hq/Hkv | D | KV |",
             "|---|---|---:|---|---|---:|---|"]

    def compact(values):
        if len(set(values)) == 1:
            return str(values[0]) + (f" × {len(values)}" if len(values) > 1 else "")
        return ",".join(map(str, values))

    for c in cases:
        batch = len(c["kv_lengths"])
        q = c.get("q_lengths", [c["q_len"]] * batch)
        lines.append(f"| {c['name']} | {c['dtype']} | {batch} | {compact(q)} | "
                     f"{c['heads']}/{c['kv_heads']} | {c['dim']} | {compact(c['kv_lengths'])} |")
    for fd in (True, False):
        for lse in (True, False):
            label = "两边均 FD" if fd else "FA 为 FD、FIA 为非 FD"
            lines += ["", f"## {label}，LSE {'on' if lse else 'off'}", "",
                      "总耗时包含每次调用的全部计算 kernel，LSE on 的 FA 包含 Fill 初始化。"
                      "耗时为三轮均值，括号是三轮均值的最小–最大值，单位 µs。"
                      "FA/FIA > 1 表示 FA 更慢。split 列是实际出现的 split 数集合。", "",
                      "| case | FA auto split | FA16 实际 split | FIA split | FA auto 总 µs | FA auto 主 kernel µs | FA16 总 µs | FIA 总 µs | FA auto/FIA |",
                      "|---|---|---|---|---:|---:|---:|---:|---:|"]
            for r in results:
                if r["fia_fd"] != fd or r["return_lse"] != lse:
                    continue
                timing = lambda v: (f"{r[v]['mean_us']:.3f} "
                                    f"({r[v]['min_round_us']:.3f}–{r[v]['max_round_us']:.3f})")
                split = lambda key: ",".join(map(str, r[key])) or "非 FD"
                lines.append(f"| {r['case']} | {split('fa0_splits')} | {split('fa16_splits')} | "
                             f"{split('fia_splits')} | {timing('fa0')} | {r['fa0']['main_mean_us']:.3f} | {timing('fa16')} | "
                             f"{timing('fia')} | {r['fa0_over_fia']:.3f} |")
    (root / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
