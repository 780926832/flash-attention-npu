"""Profile FIA with the query/head/dtype configuration of the original FA test.

The default KV length is 8192: the current Ascend950 tiling enables FD with
16 KV partials. --kv-len 2048 reproduces the original FA shape, for which the
current FIA tiling disables FD. Check tiling logs when changing the shape.

Run after loading CANN, for example:
    ASCEND_RT_VISIBLE_DEVICES=2 python tests/profile_fia_fd_combine.py --return-lse 1

Use ordinary msprof Task Duration for device timing. This script's loop includes
the requested warmup calls; discard those calls when parsing the profiler CSV.
A combine-disabled experimental binary must use --skip-check because it cannot
produce valid attention output.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu


def make_inputs(seed=1, kv_len=2048):
    if kv_len < 128 or kv_len % 128:
        raise ValueError("kv_len must be a positive multiple of 128")
    generator = torch.Generator().manual_seed(seed)

    def random_tensor(shape):
        # Match the original FA test's CPU random-number order and [-5, 5] range.
        return (-5.0 + 10.0 * torch.rand(shape, generator=generator)).bfloat16()

    q = random_tensor((1, 64, 128))
    k = random_tensor((kv_len // 128, 128, 1, 128))
    v = random_tensor(k.shape)
    digest = hashlib.sha256(b"".join(x.view(torch.uint8).numpy().tobytes() for x in (q, k, v))).hexdigest()
    return q, k, v, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-lse", type=int, choices=(0, 1), default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    q_cpu, k_cpu, v_cpu, digest = make_inputs(args.seed, args.kv_len)
    q = q_cpu.npu()
    k = k_cpu.permute(0, 2, 1, 3).contiguous().npu()
    v = v_cpu.permute(0, 2, 1, 3).contiguous().npu()
    table = torch.arange(args.kv_len // 128, dtype=torch.int32).view(1, -1).npu()

    def run():
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=64, num_key_value_heads=1, input_layout="TND",
            scale=128 ** -0.5, actual_seq_lengths=[1], actual_seq_lengths_kv=[args.kv_len],
            block_table=table, block_size=128, sparse_mode=0,
            softmax_lse_flag=bool(args.return_lse))

    print(json.dumps({"shape": f"BF16 B1 Q1 KV{args.kv_len} H64 KVH1 D128 block128 TND",
        "seed": args.seed, "input_sha256": digest, "return_lse": bool(args.return_lse),
        "warmup": args.warmup, "repeats": args.repeats,
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "default"),
        "custom_opp": os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")}), flush=True)

    result = None
    for _ in range(args.warmup + args.repeats):
        result = run()
        torch.npu.synchronize()
    if result is None:
        raise ValueError("warmup + repeats must be positive")
    output, lse = (x.cpu() for x in result)
    if not args.skip_check:
        keys = k_cpu.reshape(args.kv_len, 128).float()
        values = v_cpu.reshape(args.kv_len, 128).float()
        scores = q_cpu[0].float() @ keys.T * (128 ** -0.5)
        expected = scores.softmax(dim=-1) @ values
        actual = output.reshape(64, 128).float()
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
        metrics = {"out_max_abs": (actual - expected).abs().max().item(),
                   "out_rmse": (actual - expected).square().mean().sqrt().item()}
        if args.return_lse:
            torch.testing.assert_close(lse.reshape(64).float(), scores.logsumexp(dim=-1), atol=2e-4, rtol=2e-5)
            metrics["lse_max_abs"] = (lse.reshape(64).float() - scores.logsumexp(dim=-1)).abs().max().item()
        assert torch.isfinite(output).all()
        print("FP32 reference check passed", flush=True)
        print(json.dumps({"accuracy": metrics}), flush=True)
    if args.save_output:
        torch.save({"output": output, "lse": lse, "input_sha256": digest}, args.save_output)
    print(json.dumps({"completed": args.warmup + args.repeats,
        "output_shape": list(output.shape), "lse_shape": list(lse.shape),
        "accuracy_checked": not args.skip_check}), flush=True)


if __name__ == "__main__":
    main()
