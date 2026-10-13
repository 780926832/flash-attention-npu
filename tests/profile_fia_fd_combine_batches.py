"""Measure FIA on exactly the inputs used by profile_fd_combine_batches.py.

FIA chooses its own split and row schedules; FA case names do not assert FIA
row counts. Verify FIA tiling logs before interpreting combine timings.
Use an isolated custom package for --variant no-combine; its outputs are invalid.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import torch
import torch_npu

from profile_fd_combine_batches import CASES, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("full", "no-combine"), default="full")
    parser.add_argument("--mode", choices=("validate", "profile"), default="validate")
    parser.add_argument("--cases", type=Path, help="JSON list or an object containing a cases list")
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--return-lse", type=int, choices=(0, 1))
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-device-occupancy", action="store_true",
                        help="Require no foreign processes immediately before/after benchmark calls")
    args = parser.parse_args()
    if args.variant == "no-combine" and not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        parser.error("no-combine requires an isolated custom package via ASCEND_CUSTOM_OPP_PATH")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    torch.set_num_threads(4)
    candidates = json.loads(args.cases.read_text()) if args.cases else CASES
    if isinstance(candidates, dict):
        candidates = candidates["cases"]
    cases = [c for c in candidates if args.case is None or c["name"] in args.case]
    if not cases or (args.case and set(args.case) - {c["name"] for c in cases}):
        parser.error("empty selection or unknown case name")
    modes = (True, False) if args.return_lse is None else (bool(args.return_lse),)
    prepared = {}
    for case in cases:
        fa, expected, expected_lse, digest = make_inputs(case)
        inputs = dict(query=fa["q"], key=fa["k_cache"].permute(0, 2, 1, 3).contiguous(),
                      value=fa["v_cache"].permute(0, 2, 1, 3).contiguous(),
                      num_heads=case["heads"], num_key_value_heads=case.get("kv_heads", 1), input_layout="TND",
                      scale=case["dim"] ** -0.5, actual_seq_lengths=fa["cu_seqlens_q"].cpu().tolist()[1:],
                      actual_seq_lengths_kv=case["kv_lengths"], block_table=fa["page_table"],
                      block_size=128, sparse_mode=0)
        prepared[case["name"]] = inputs, expected, expected_lse, digest
    groups = [dict(case=c["name"], return_lse=lse) for c in cases for lse in modes]
    metadata = dict(variant=args.variant, physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    device_name=torch.npu.get_device_name(0),
                    custom_opp=os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""), groups=groups,
                    fa_reference_cases=cases,
                    input_sha256={name: x[3] for name, x in prepared.items()},
                    warmup=args.warmup if args.mode == "profile" else 0,
                    measured=args.iterations if args.mode == "profile" else 1)
    print(json.dumps(metadata), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check_occupancy(label):
        if not args.check_device_occupancy:
            return
        device = int(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
        snapshot = subprocess.check_output(["npu-smi", "info"], text=True)
        args.output.with_name(f"device-{label}-calls.log").write_text(snapshot)
        active = {int(pid) for card, pid in re.findall(
            r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', snapshot, re.MULTILINE) if int(card) == device}
        if os.getpid() not in active:
            raise RuntimeError("Cannot confirm benchmark process in device occupancy snapshot")
        if active - {os.getpid()}:
            raise RuntimeError("FOREIGN_NPU_PROCESS: " + str(sorted(active - {os.getpid()})))

    check_occupancy("before")
    outputs, checks = {}, []
    calls = args.warmup + args.iterations if args.mode == "profile" else 1
    for case in cases:
        inputs, expected, expected_lse, _ = prepared[case["name"]]
        mode_outputs = []
        for lse in modes:
            label = f"{case['name']}-lse{int(lse)}"
            print(json.dumps(dict(case_begin=case["name"], return_lse=lse)), flush=True)
            for _ in range(calls):
                result = torch_npu.npu_fused_infer_attention_score(**inputs, softmax_lse_flag=lse)
                torch.npu.synchronize()
            if args.variant == "full":
                out = result[0].cpu()
                atol, rtol = (.005, .005) if case.get("dtype") == "fp16" else (.03, .02)
                torch.testing.assert_close(out.float(), expected, atol=atol, rtol=rtol)
                err = out.float() - expected
                metrics = dict(out_max_abs=err.abs().max().item(), out_rmse=err.square().mean().sqrt().item())
                tensors = {"out": out}
                if lse:
                    raw_lse = result[1].cpu()
                    assert tuple(raw_lse.shape) == (*expected.shape[:2], 1), raw_lse.shape
                    actual_lse = raw_lse.squeeze(-1).T.contiguous()
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
        if len(mode_outputs) == 2:
            assert torch.equal(*mode_outputs), case["name"]
    check_occupancy("after")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata=metadata, outputs=outputs, accuracy=checks), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        dict(metadata=metadata, accuracy=checks), indent=2) + "\n")
    print(f"Completed {len(groups) * calls} attention calls.", flush=True)


if __name__ == "__main__":
    main()
