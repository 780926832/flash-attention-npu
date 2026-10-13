"""Extract actual FIA FD schedules from validation stdout with CANN INFO logs.

Input must contain profile_fia_fd_combine_batches.py's case_begin markers.
Metrics describe source-level logical work, not measured hardware traffic.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def parse_log(path):
    keywords = ("fdRes.", "tilingkey param", "Tiling outputs", "bin path is")
    with path.open(errors="replace") as file:
        text = "".join(line for line in file if line.startswith('{"') or
                       ("FusedInferAttentionScore" in line and any(k in line for k in keywords)))
    decode = lambda line: json.JSONDecoder().raw_decode(line)[0]
    metadata = next(decode(line) for line in text.splitlines()
                    if line.startswith('{"variant":'))
    cases = {c["name"]: c for c in metadata["fa_reference_cases"]}
    blocks = re.split(r'(?m)^(?=\{"case_begin":)', text)[1:]
    records = []
    for block in blocks:
        marker = decode(block.splitlines()[0])
        name = marker.pop("case_begin")
        record = dict(case=name, **marker, variant=metadata["variant"])
        keys = re.findall(r'tilingKey: (\d+), numBlocks: (\d+)', block)
        flags = re.findall(r'isFd: (\d+)', block)
        record["is_fd"] = bool(flags and int(flags[-1]))
        if keys:
            record.update(tiling_key=int(keys[-1][0]), aic=int(keys[-1][1]),
                          aiv=2 * int(keys[-1][1]))
        checks = [decode(line) for line in block.splitlines() if line.startswith('{"case":')]
        assert len(checks) == 1, (name, len(checks))
        record["check"] = checks[0]
        if record["is_fd"]:
            arrays = {}
            for field in ("fdS2SplitNum", "fdBN2Idx", "fdMIdx", "fdWorkspaceIdx",
                          "mSize", "taskIdx", "mStart", "mLen"):
                values = re.findall(r'fdRes\.' + field + r'\[(\d+)\]: (\d+)', block)
                arrays[field] = [int(v) for _, v in sorted(values, key=lambda x: int(x[0]))]
            record["arrays"] = arrays
            record["fd_num"] = int(re.search(r'fdRes.fdNum: (\d+)', block)[1])
            record["fd_vec"] = int(re.search(r'fdRes.fdUsedVecNum: (\d+)', block)[1])
            assert len(arrays["taskIdx"]) == len(arrays["mLen"]) == record["fd_vec"]
            assert len(arrays["mSize"]) == len(arrays["fdS2SplitNum"]) == record["fd_num"]
            cores, tiles = [], Counter()
            coverage = [Counter() for _ in arrays["mSize"]]
            for idx, (task, start, length) in enumerate(zip(
                    arrays["taskIdx"], arrays["mStart"], arrays["mLen"])):
                split = arrays["fdS2SplitNum"][task]
                assert split >= 2
                coverage[task].update(range(start, start + length))
                rounds = math.ceil(length / 8)
                for row in range(0, length, 8):
                    tiles[(split, min(8, length - row))] += 1
                cores.append(dict(aiv=idx, task=task, rows=length, splits=split,
                                  row_rounds=rounds, o_dma_calls=rounds * split,
                                  o_read_KiB=length * split * cases[name]["dim"] * 4 / 1024))
            for task, counts in enumerate(coverage):
                assert counts == dict.fromkeys(range(arrays["mSize"][task]), 1), (name, task)
            record["metrics"] = dict(
                split_set=sorted(set(arrays["fdS2SplitNum"])),
                row_tile_set=sorted({row for split, row in tiles}),
                active_aiv=len(cores), launched_aiv=record["aiv"],
                max_row_rounds=max(c["row_rounds"] for c in cores),
                max_o_dma_calls=max(c["o_dma_calls"] for c in cores),
                max_o_read_KiB=max(c["o_read_KiB"] for c in cores),
                total_o_read_KiB=sum(c["o_read_KiB"] for c in cores),
                total_o_dma_calls=sum(c["o_dma_calls"] for c in cores),
                tiles=[dict(splits=s, rows=r, count=n) for (s, r), n in sorted(tiles.items())],
                cores=cores)
            binary_match = re.search(r'bin path is (\S+\.o)', block)
            if binary_match:
                binary = Path(binary_match[1]).resolve()
                record.update(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                if metadata["custom_opp"]:
                    assert binary.is_relative_to(Path(metadata["custom_opp"]).resolve())
        records.append(record)
    assert len(records) == len(metadata["groups"])
    return dict(metadata=metadata, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = parse_log(args.log)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(dict(groups=len(data["records"]),
                          fd_groups=sum(r["is_fd"] for r in data["records"]))))


if __name__ == "__main__":
    main()
