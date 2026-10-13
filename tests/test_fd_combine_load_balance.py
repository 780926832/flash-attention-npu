# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
"""FD scheduling regressions with independent, nonuniform FP32 references."""

import pytest
import torch
import torch_npu

from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_forward


CASES = {
    "weighted_d128": ([14] * 4, None, [8192, 2048, 2048, 2048], 8, 1, 128),
    "weighted_d256": ([16] * 7, None, [8192] + [1024] * 6, 4, 1, 256),
    "source_tail_d127": ([10] * 4, None, [8192, 2048, 2048, 2048], 8, 1, 127),
    "source_tail_d255": ([15] * 7, None, [8192] + [1024] * 6, 4, 1, 255),
    "varlen_head_tail": ([3, 2, 1], None, [4096, 2048, 512], 10, 2, 65),
    # Used Q lengths determine task geometry, but actual cuQ determines output
    # addresses. Reconstructing addresses from used-length prefixes is wrong.
    "used_q_storage_gaps": ([18, 14, 13, 17], [14, 11, 9, 16],
                            [8192, 2048, 2048, 2048], 8, 1, 127),
}


def run_reference_case(name, dtype, splits, return_lse):
    q_lengths, used_q, kv_lengths, heads, kv_heads, dim = CASES[name]
    batch, page = len(q_lengths), 128
    cu_q = torch.tensor([0] + q_lengths, dtype=torch.int32).cumsum(0).int()
    used = q_lengths if used_q is None else used_q
    max_kv = max(kv_lengths)
    q = torch.zeros((int(cu_q[-1]), heads, dim), dtype=dtype)
    q[..., 0] = ((torch.arange(q.shape[0] * heads).reshape(-1, heads) % 13 - 6) / 8).to(dtype)
    k = torch.zeros((batch, max_kv, kv_heads, dim), dtype=dtype)
    logits = ((torch.arange(max_kv) // page) % 9 - 4).float() * 2
    k[..., 0] = (logits[None, :, None] + torch.arange(kv_heads)[None, None, :] / 2).to(dtype)
    gen = torch.Generator().manual_seed(20261013)
    v = (torch.rand(k.shape, generator=gen) * 2 - 1).to(dtype)
    v[..., 0] = 1
    expected, expected_lse, valid_rows = [], [], []
    for bi, length in enumerate(used):
        start = int(cu_q[bi])
        if not length:
            continue
        valid_rows.extend(range(start, start + length))
        query = q[start:start + length].float().transpose(0, 1)
        key = k[bi, :kv_lengths[bi]].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        value = v[bi, :kv_lengths[bi]].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        # Only component zero is nonzero in Q and K, so scores are exact in
        # FP32. This checks combining unequal softmax mass, not just a mean.
        scores = query @ key.transpose(1, 2)
        expected.append((scores.softmax(-1) @ value).transpose(0, 1))
        expected_lse.append(scores.logsumexp(-1))
    optional = {key: None for key in (
        "k_new", "v_new", "qv", "out", "cu_seqlens_k", "cu_seqlens_k_new",
        "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin", "seqlens_rotary",
        "q_descale", "k_descale", "v_descale", "scheduler_metadata", "pack_gqa")}
    result = _flash_attn_forward(
        q=q.npu(), k=k.reshape(-1, page, kv_heads, dim).npu(),
        v=v.reshape(-1, page, kv_heads, dim).npu(), cu_seqlens_q=cu_q.npu(),
        seqused_q=None if used_q is None else torch.tensor(used_q, dtype=torch.int32).npu(),
        seqused_k=torch.tensor(kv_lengths, dtype=torch.int32).npu(),
        max_seqlen_q=max(q_lengths), max_seqlen_k=max_kv,
        page_table=torch.arange(batch * max_kv // page, dtype=torch.int32).reshape(batch, -1).npu(),
        softmax_scale=1.0, causal=False, window_size_left=-1, window_size_right=-1,
        attention_chunk=0, softcap=0.0, rotary_interleaved=True, num_splits=splits,
        sm_margin=0, return_softmax_lse=return_lse, **optional,
    )
    out = result[0].cpu()[valid_rows]
    ref = torch.cat(expected)
    torch.testing.assert_close(out.float(), ref,
                               atol=2e-4 if dtype == torch.bfloat16 else 3e-5,
                               rtol=8e-3 if dtype == torch.bfloat16 else 2e-3)
    torch.testing.assert_close(out[..., 0], torch.ones_like(out[..., 0]), atol=0, rtol=0)
    tensors = {"out": out}
    metrics = {"out_max_abs": (out.float() - ref).abs().max().item()}
    if return_lse:
        lse = result[1].cpu()[:, valid_rows]
        ref_lse = torch.cat(expected_lse, 1)
        torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=0)
        tensors["lse"] = lse
        metrics["lse_max_abs"] = (lse - ref_lse).abs().max().item()
    return tensors, metrics


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [0, 16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fd_load_balance_reference(name, dtype, splits, return_lse):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")
    run_reference_case(name, dtype, splits, return_lse)
