"""Profile a JSON matrix of natural FA FD tilings with isolated native variants.

Use the accompanying dated report for the full/no-combine measurement method.
The disabled variant has invalid outputs; it is never used for accuracy checks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import torch_npu

from profile_fd_combine_batches import make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", choices=("validate", "profile"), default="validate")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-device-occupancy", action="store_true")
    parser.add_argument("--occupancy-wait", type=float, default=120,
                        help="Seconds to wait for foreign processes before benchmark calls")
    args = parser.parse_args()
    cases = json.loads(args.cases_json.read_text())
    native_record = json.loads(args.manifest.read_text())[args.variant]
    combine_disabled = native_record.get("combine_disabled", args.variant == "no-combine")
    sys.path.insert(0, str(Path(native_record["native"]).parent))
    import flash_attn_npu_3_950 as native
    from flash_attn_npu_3 import flash_attn_with_kvcache

    native_path = Path(native.__file__).resolve()
    assert native_path == Path(native_record["native"]).resolve()
    native_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native_sha == native_record["sha256"]
    torch.set_num_threads(4)
    assert torch.npu.get_device_properties(0).cube_core_num == 28
    prepared = {c["name"]: make_inputs(c) for c in cases}
    for case in cases:
        prepared[case["name"]][0]["num_splits"] = case["num_splits"]
    groups = [dict(case=c["name"], return_lse=lse, aic=c["aic"], aiv=2 * c["aic"])
              for c in cases for lse in (True, False)]
    metadata = dict(variant=args.variant, native=str(native_path), native_sha256=native_sha,
                    physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    groups=groups, cases=cases,
                    input_sha256={k: v[3] for k, v in prepared.items()},
                    warmup=args.warmup if args.mode == "profile" else 0,
                    measured=args.iterations if args.mode == "profile" else 1)
    print(json.dumps(metadata), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check_occupancy(label):
        if not args.check_device_occupancy:
            return
        device = int(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
        started = time.monotonic()
        while True:
            snapshot = subprocess.check_output(["npu-smi", "info"], text=True)
            args.output.with_name(f"device-{label}-calls.log").write_text(snapshot)
            active = {int(pid) for card, pid in re.findall(
                r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', snapshot, re.MULTILINE) if int(card) == device}
            if os.getpid() not in active:
                raise RuntimeError("Cannot confirm benchmark process in device occupancy snapshot")
            if not active - {os.getpid()}:
                return
            if label != "before" or time.monotonic() - started >= args.occupancy_wait:
                raise RuntimeError("FOREIGN_NPU_PROCESS: " + str(sorted(active - {os.getpid()})))
            time.sleep(1)

    check_occupancy("before")
    outputs, checks = {}, []
    calls = metadata["warmup"] + metadata["measured"]
    for case in cases:
        inputs, expected, expected_lse, _ = prepared[case["name"]]
        mode_outputs = []
        for lse in (True, False):
            label = f"{case['name']}-lse{int(lse)}"
            print(json.dumps(dict(case_begin=case["name"], return_lse=lse)), flush=True)
            for _ in range(calls):
                result = flash_attn_with_kvcache(**inputs, return_softmax_lse=lse)
                torch.npu.synchronize()
            if not combine_disabled:
                out = (result[0] if lse else result).cpu()
                atol, rtol = ((.005, .005) if case.get("dtype") == "fp16" else (.03, .02))
                torch.testing.assert_close(out.float(), expected, atol=atol, rtol=rtol)
                err = out.float() - expected
                metrics = dict(out_max_abs=err.abs().max().item(),
                               out_rmse=err.square().mean().sqrt().item())
                tensors = {"out": out}
                if lse:
                    actual_lse = result[1].cpu()
                    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-4, rtol=2e-5)
                    metrics["lse_max_abs"] = (actual_lse - expected_lse).abs().max().item()
                    tensors["lse"] = actual_lse
                outputs[label] = tensors
                mode_outputs.append(out)
            else:
                metrics = dict(skipped="combine disabled: invalid outputs are not read")
            record = dict(case=label, accuracy=metrics)
            checks.append(record)
            print(json.dumps(record), flush=True)
        if mode_outputs:
            assert torch.equal(*mode_outputs), case["name"]
    check_occupancy("after")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata=metadata, outputs=outputs, accuracy=checks), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        dict(metadata=metadata, accuracy=checks), indent=2) + "\n")
    print(f"Completed {len(groups) * calls} attention calls.", flush=True)


if __name__ == "__main__":
    main()
