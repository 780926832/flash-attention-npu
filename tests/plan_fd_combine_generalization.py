"""Generate one-variable FD sweeps and inspect actual host combine schedules.

Build fd_combine_tiling_probe.cpp as a shared library, then pass --probe and
--output-dir. Device validation must still confirm these host predictions.
"""

import argparse
import collections
import ctypes as C
import json
import math
from pathlib import Path


def cases():
    result = []

    def add(family, value, **kwargs):
        result.append(dict(name=f"{family}_{value}", family=family, value=value,
                           **(dict(q_len=14, heads=8, kv_heads=1, dim=128,
                                   kv_lengths=[8192, 2048, 2048, 2048],
                                   num_splits=16, dtype="bf16") | kwargs)))

    for q in range(1, 17):
        add("q", q, q_len=q)
    for d in (32, 64, 65, 96, 127, 128, 129, 160, 192, 224, 255, 256):
        add("d", d, q_len=16, heads=4, dim=d,
            kv_lengths=[8192] + [1024] * 6)
    for h in (8, 16, 24, 32, 40, 48, 64, 96, 128):
        add("heads", h, q_len=1, heads=h, kv_lengths=[8192])
    for s in (0, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28):
        add("split_single", s, q_len=1, heads=64, kv_lengths=[8192], num_splits=s)
        add("split_mixed", s, num_splits=s)
    for kv in (1024, 1025, 2048, 4096, 8191, 8192, 8193, 16384, 32768):
        add("kv", kv, q_len=1, heads=64, kv_lengths=[kv])
    for b in range(1, 10):
        add("batch", b, q_len=4, kv_lengths=[8192] * b)
    for aux in (128, 256, 512, 1024, 2048, 4096, 8192):
        add("skew", aux, kv_lengths=[8192] + [aux] * 3)
    for kh in (1, 2, 4, 8):
        add("kvheads", kh, q_len=1, heads=64, kv_heads=kh, kv_lengths=[8192])
    for name, q, kv in (
        ("ascending", [1, 4, 8, 16], [8192, 2048, 2048, 2048]),
        ("descending", [16, 8, 4, 1], [8192, 2048, 2048, 2048]),
        ("matched", [1, 4, 8, 16], [1024, 2048, 4096, 8192]),
        ("tails", [3, 7, 11, 15], [8193, 1025, 2049, 4097]),
    ):
        add("varq", name, q_lengths=q, kv_lengths=kv)
    for name, q, h, kh, kv in (
        ("d256", 16, 4, 1, [8192] + [1024] * 6),
        ("gqa", 1, 64, 4, [8192]),
    ):
        add("auto", name, q_len=q, heads=h, kv_heads=kh, kv_lengths=kv,
            dim=256 if name == "d256" else 128, num_splits=0)
    for name, q, h, d, kv in (
        ("single", 1, 64, 128, [8192]),
        ("mixed", 14, 8, 128, [8192, 2048, 2048, 2048]),
        ("d256", 16, 4, 256, [8192] + [1024] * 6),
        ("tail", 16, 4, 255, [8192] + [1024] * 6),
    ):
        add("fp16", name, q_len=q, heads=h, dim=d, kv_lengths=kv, dtype="fp16")
    return result


def load_probe(path):
    lib = C.CDLL(str(path.resolve()))
    lib.tiling_size.restype = C.c_size_t
    lib.probe_raw.argtypes = [C.c_int, C.POINTER(C.c_int32), C.c_int, C.c_int,
                             C.c_int, C.POINTER(C.c_int32), C.c_int, C.c_int, C.c_void_p]
    lib.describe_tiling.argtypes = [C.c_void_p]
    lib.describe_tiling.restype = C.c_char_p
    return lib


def probe_case(lib, case):
    kv = case["kv_lengths"]
    q = case.get("q_lengths", [case["q_len"]] * len(kv))
    raw = C.create_string_buffer(lib.tiling_size())
    lib.probe_raw(len(kv), (C.c_int32 * len(q))(*q), case["heads"],
                  case["kv_heads"], case["dim"], (C.c_int32 * len(kv))(*kv),
                  28, case["num_splits"], raw)
    return json.loads(lib.describe_tiling(raw))


def schedule_metrics(plan, lse):
    """Exact host-derived work counts, not hardware counters or GB/s."""
    rows = [x[4] for x in plan["task_info"]]
    uniform = bool(plan["uniform_modes"] & (2 if lse else 1))
    cores = [[] for _ in range(plan["aiv"])]
    if uniform:
        tile = 0
        for task, n in enumerate(rows):
            split = min(n, (n + 7) // 8 * 4)
            for sub, size in enumerate((split, n - split)):
                for start in range(0, size, plan["row_tile"]):
                    cores[tile % len(cores)].append(
                        (task, sub, start, min(plan["row_tile"], size - start)))
                    tile += 1
    else:
        cores = [[s] if s[3] else [] for s in plan["core_schedules"]]
    details, coverage = [], [collections.Counter() for _ in rows]
    for i, assignments in enumerate(cores):
        rounds, read, dma, max_dma, dma_calls, extents = 0, 0, 0, 0, 0, []
        for task, sub, start, count in assignments:
            partials, dim = plan["schedules"][task]["partials"], plan["D"]
            split = min(rows[task], (rows[task] + 7) // 8 * 4)
            first = (split if sub else 0) + start
            coverage[task].update(range(first, first + count))
            read += count * partials * dim * 4
            for offset in range(0, count, 8):
                extent = min(8, count - offset)
                capacity = 8192 // (extent * ((dim + 7) // 8 * 8))
                batches = math.ceil(partials / capacity)
                rounds += 1
                dma += batches
                max_dma = max(max_dma, batches)
                dma_calls += partials if dim % 8 else batches
                extents.append(extent)
        details.append(dict(aiv=i, rounds=rounds, bytes=read, dma_batches=dma,
                            max_dma_per_round=max_dma, o_dma_calls=dma_calls,
                            row_extents=extents))
    for i, counts in enumerate(coverage):
        assert counts == dict.fromkeys(range(rows[i]), 1)
    return dict(mode="uniform" if uniform else "balanced",
                active=sum(bool(x) for x in cores), launched=plan["aiv"],
                total_KiB=sum(x["bytes"] for x in details) / 1024,
                max_KiB=max(x["bytes"] for x in details) / 1024,
                max_rounds=max(x["rounds"] for x in details),
                max_dma_batches=max(x["dma_batches"] for x in details),
                max_dma_per_round=max(x["max_dma_per_round"] for x in details),
                max_o_dma_calls=max(x["o_dma_calls"] for x in details),
                total_o_dma_calls=sum(x["o_dma_calls"] for x in details),
                row_extents=sorted({v for x in details for v in x["row_extents"]}),
                cores=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lib, accepted, rejected, plans = load_probe(args.probe), [], [], []
    for case in cases():
        plan = probe_case(lib, case)
        if not plan["flash_decode"]:
            rejected.append(dict(case=case, host=plan))
            continue
        case.update(aic=plan["aic"], partials=[s["partials"] for s in plan["schedules"]])
        accepted.append(case)
        plans.append(dict(case=case["name"], host=plan,
                          metrics={str(int(lse)): schedule_metrics(plan, lse)
                                   for lse in (True, False)}))
    for name, data in (("cases.json", accepted), ("excluded-cases.json", rejected),
                       ("schedules.json", plans)):
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    print(f"FD candidates: {len(accepted)}, excluded: {len(rejected)}")


if __name__ == "__main__":
    main()
