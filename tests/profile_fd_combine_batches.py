"""Exercise natural FD combine row tiles and one/two/four partial-load batches.

The two rows8 cases use auxiliary variable-length batches to trigger rowTile=8.
Only batch 0 has 16 actual partials; timings cover the whole combine stage.
Use --manifest with the saved full/no-combine binaries for an A/B experiment.
The no-combine variant produces invalid outputs and is used only for timing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

import torch
import torch_npu


CASES = [
    dict(name="rows2_d128", q_len=1, heads=64, dim=128,
         kv_lengths=[8192], row_tile=2, partials=[16], aic=16),
    dict(name="rows8_d128", q_len=14, heads=8, dim=128,
         kv_lengths=[8192, 2048, 2048, 2048], row_tile=8,
         partials=[16, 4, 4, 4], aic=28),
    dict(name="rows8_d256", q_len=16, heads=4, dim=256,
         kv_lengths=[8192, 1024, 1024, 1024, 1024, 1024, 1024], row_tile=8,
         partials=[16, 2, 2, 2, 2, 2, 2], aic=28),
]


def make_inputs(case):
    gen = torch.Generator().manual_seed(1)
    batch, q_len, heads, dim = len(case["kv_lengths"]), case["q_len"], case["heads"], case["dim"]
    pages = max(case["kv_lengths"]) // 128

    def rand(shape):
        return (-5 + 10 * torch.rand(shape, generator=gen)).bfloat16()

    q_cpu = rand((batch * q_len, heads, dim))
    k_cpu = rand((batch * pages, 128, 1, dim))
    v_cpu = rand(k_cpu.shape)
    fingerprint = hashlib.sha256(b"".join(
        x.view(torch.uint8).numpy().tobytes() for x in (q_cpu, k_cpu, v_cpu))).hexdigest()
    if case["name"] == "rows2_d128":
        assert fingerprint == "3a556ea7c526c920233a2ec8a07f180acb6de6bb9e43ec0bded28abb0435b70f"
    references, lses = [], []
    for bi, kv_len in enumerate(case["kv_lengths"]):
        q = q_cpu[bi * q_len:(bi + 1) * q_len].float().reshape(-1, dim)
        k = k_cpu[bi * pages:(bi + 1) * pages].reshape(-1, dim)[:kv_len].float()
        v = v_cpu[bi * pages:(bi + 1) * pages].reshape(-1, dim)[:kv_len].float()
        scores = (q @ k.T) * dim ** -0.5
        references.append((scores.softmax(-1) @ v).reshape(q_len, heads, dim))
        lses.append(scores.logsumexp(-1).reshape(q_len, heads).T)
    inputs = dict(q=q_cpu.npu(), k_cache=k_cpu.npu(), v_cache=v_cpu.npu(),
                  cache_seqlens=torch.tensor(case["kv_lengths"], dtype=torch.int32, device="npu"),
                  page_table=torch.arange(batch * pages, dtype=torch.int32, device="npu").reshape(batch, pages),
                  cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device="npu") * q_len,
                  max_seqlen_q=q_len, softmax_scale=dim ** -0.5, causal=False,
                  window_size=(-1, -1), softcap=0., num_splits=16)
    return inputs, torch.cat(references), torch.cat(lses, dim=1), fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("full", "no-combine"), default="full")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("validate", "profile"), default="validate")
    parser.add_argument("--case", choices=[c["name"] for c in CASES])
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.variant == "no-combine" and args.manifest is None:
        parser.error("no-combine requires an explicit isolated binary manifest")
    manifest = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())[args.variant]
        sys.path.insert(0, str(Path(manifest["native"]).parent))
    import flash_attn_npu_3_950 as native
    from flash_attn_npu_3 import flash_attn_with_kvcache

    native_path = Path(native.__file__).resolve()
    native_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
    if manifest:
        assert native_path == Path(manifest["native"]).resolve()
        assert native_sha == manifest["sha256"]
    torch.set_num_threads(4)
    assert torch.npu.get_device_properties(0).cube_core_num == 28, "Cases target a 28-AIC Ascend950"
    cases = [c for c in CASES if args.case is None or c["name"] == args.case]
    prepared = {c["name"]: make_inputs(c) for c in cases}
    groups = [dict(case=c["name"], return_lse=lse, aic=c["aic"], aiv=2 * c["aic"])
              for c in cases for lse in (True, False)]
    metadata = dict(variant=args.variant, native=str(native_path), native_sha256=native_sha,
                    physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    device_name=torch.npu.get_device_name(0), groups=groups, cases=cases,
                    input_sha256={name: value[3] for name, value in prepared.items()},
                    warmup=args.warmup if args.mode == "profile" else 0,
                    measured=args.iterations if args.mode == "profile" else 1)
    print(json.dumps(metadata), flush=True)
    outputs, checks = {}, []
    calls = args.warmup + args.iterations if args.mode == "profile" else 1
    for case in cases:
        inputs, expected, expected_lse, _ = prepared[case["name"]]
        mode_outputs = []
        for lse in (True, False):
            label = f"{case['name']}-lse{int(lse)}"
            for _ in range(calls):
                result = flash_attn_with_kvcache(**inputs, return_softmax_lse=lse)
                torch.npu.synchronize()
            if args.variant == "full":
                out = (result[0] if lse else result).cpu()
                torch.testing.assert_close(out.float(), expected, atol=.03, rtol=.02)
                err = out.float() - expected
                metrics = dict(out_max_abs=err.abs().max().item(), out_rmse=err.square().mean().sqrt().item())
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata=metadata, outputs=outputs, accuracy=checks), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        dict(metadata=metadata, accuracy=checks), indent=2) + "\n")
    print(f"Completed {len(groups) * calls} attention calls.", flush=True)


if __name__ == "__main__":
    main()
