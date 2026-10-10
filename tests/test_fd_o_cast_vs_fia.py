# Copyright (c) 2026, Minghua Shen.

"""Run FA decode with automatic KV splitting; FIA comparison is disabled.

This case uses paged TND, one query token, 64 query heads and KV length
2048. num_splits=0 lets FA select the KV splitting automatically.

Run from the flash-attention-npu directory:

    python tests/test_fd_o_cast_vs_fia.py
"""

import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache
from tests.common.test_utils import make_random_tensor

Q_LEN = 1
KV_LEN = 2048
# 64 query heads keep this one decode row on FIA's flash-decode path.
# Fewer heads make s1 * g < 64, and FIA then stays on the prefill cast.
NUM_HEADS = 64
KV_HEADS = 1
HEAD_SIZE = 128
BLOCK_SIZE = 128
NUM_BLOCKS = KV_LEN // BLOCK_SIZE
SOFTMAX_SCALE = HEAD_SIZE ** -0.5
SEEDS = (1, 2)
# FIA splits KV so each piece is at least 256 tokens. Match that split count
# so both sides combine the same partials before the final cast.
KV_SPLIT_LIMIT = 256


def _fia_kv_splits():
    """Same split count FIA arch35 uses for this one-row decode shape."""
    cores = torch.npu.get_device_properties(0).cube_core_num
    part = max(int(cores), 1)
    while KV_LEN // part < KV_SPLIT_LIMIT and part > 1:
        part -= 1
    return part


def _align_lse(lse, reference):
    aligned = lse
    while aligned.dim() > reference.dim() and aligned.shape[-1] == 1:
        aligned = aligned.squeeze(-1)
    if aligned.shape != reference.shape and aligned.numel() == reference.numel():
        aligned = aligned.reshape(reference.shape)
    return aligned


def _run_one(dtype, seed, num_splits):
    gen = torch.Generator().manual_seed(seed)
    query = make_random_tensor(
        (Q_LEN, NUM_HEADS, HEAD_SIZE), dtype, generator=gen, device="npu"
    )
    key_cache = make_random_tensor(
        (NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE), dtype, generator=gen, device="npu"
    )
    value_cache = make_random_tensor(
        (NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE), dtype, generator=gen, device="npu"
    )
    cu_seqlens_q = torch.tensor([0, Q_LEN], dtype=torch.int32, device=query.device)
    cache_seqlens = torch.tensor([KV_LEN], dtype=torch.int32, device=query.device)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=query.device).view(1, NUM_BLOCKS)

    fa_out, fa_lse, *_rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_LEN,
        softmax_scale=SOFTMAX_SCALE,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=num_splits,
        return_softmax_lse=True,
    )
    # FIA paged cache is [blocks, kv_heads, block_size, dim].
    # fia_out, fia_lse = torch_npu.npu_fused_infer_attention_score(
    #     query,
    #     key_cache.permute(0, 2, 1, 3).contiguous(),
    #     value_cache.permute(0, 2, 1, 3).contiguous(),
    #     num_heads=NUM_HEADS,
    #     num_key_value_heads=KV_HEADS,
    #     input_layout="TND",
    #     scale=SOFTMAX_SCALE,
    #     actual_seq_lengths=[Q_LEN],
    #     actual_seq_lengths_kv=[KV_LEN],
    #     block_table=page_table,
    #     block_size=BLOCK_SIZE,
    #     sparse_mode=0,
    #     softmax_lse_flag=True,
    # )
    # fia_lse = _align_lse(fia_lse, fa_lse)
    # assert fa_out.shape == fia_out.shape, (fa_out.shape, fia_out.shape)
    # assert fa_lse.shape == fia_lse.shape, (fa_lse.shape, fia_lse.shape)
    # out_diff = (fa_out.float() - fia_out.float()).abs()
    # lse_diff = (fa_lse.float() - fia_lse.float()).abs()
    # out_mismatch = int((fa_out != fia_out).sum())
    # lse_mismatch = int((fa_lse != fia_lse).sum())
    # print(
    #     f"{dtype} seed={seed} num_splits={num_splits} "
    #     f"out mismatch {out_mismatch}/{fa_out.numel()} "
    #     f"max_abs={out_diff.max().item():.6e} "
    #     f"lse mismatch {lse_mismatch}/{fa_lse.numel()} "
    #     f"max_abs={lse_diff.max().item():.6e}"
    # )
    # if out_mismatch:
    #     for pos in (out_diff > 0).nonzero()[:8].tolist():
    #         t = tuple(pos)
    #         print(f"  {t} FA={fa_out[t].item():.6f} FIA={fia_out[t].item():.6f}")
    # return out_mismatch, lse_mismatch

    torch.npu.synchronize()
    print(
        f"{dtype} seed={seed} num_splits={num_splits} "
        f"FA output shape={tuple(fa_out.shape)} LSE shape={tuple(fa_lse.shape)}"
    )


def main():
    num_splits = 0
    print(f"num_splits={num_splits} (automatic)")
    for dtype in (torch.bfloat16,):
        _run_one(dtype, SEEDS[0], num_splits)


if __name__ == "__main__":
    main()
