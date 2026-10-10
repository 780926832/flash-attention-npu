# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,q_len,heads,kv_heads,dim,splits", [
    (1, 1, 64, 1, 128, 0),  # Decode target: several AIV row tiles.
    (2, 1, 8, 1, 128, 4),   # LSE head stride with multiple batches.
    (1, 3, 10, 2, 192, 3),  # Tail head blocks and tiles crossing heads.
    (2, 3, 6, 1, 59, 4),    # Packed partial O and TND output strides.
    (1, 16, 8, 1, 64, 4),   # Maximum merged Q-head row count.
    (1, 1, 1, 1, 1, 2),     # Only one source AIV has rows.
    (1, 1, 8, 1, 256, 2),   # Maximum supported head dimension.
    (2, 1, 5, 1, 59, 4),    # Uneven source AIV row ranges.
    (2, 3, 6, 1, 65, 4),    # Different FP32 input and low-precision output strides.
    (2, 1, 5, 1, 1, 4),     # Multi-row FP32 input stride of one 32-byte block.
])
def test_fd_combine_output_and_lse(dtype, batch, q_len, heads, kv_heads, dim, splits):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")

    # Every key is identical, so attention is uniform. Different scores per
    # token/head and values per batch/KV head expose incorrect output scattering.
    total_q = batch * q_len
    q = torch.zeros((total_q, heads, dim), dtype=dtype)
    scores = torch.arange(total_q * heads).reshape(total_q, heads).float() / 128
    q[..., 0] = scores.to(dtype)
    kv_len, page_size = 2048, 128
    pages_per_batch = kv_len // page_size
    k = torch.ones((batch * pages_per_batch, page_size, kv_heads, dim), dtype=dtype)
    values = (torch.arange(batch * kv_heads).reshape(batch, kv_heads, 1).float() + 1
              + (torch.arange(dim).float() % 4) / 4).to(dtype)
    v = values[:, None, None].expand(batch, pages_per_batch, page_size, kv_heads, dim)
    v = v.reshape_as(k).contiguous()
    table = torch.arange(batch * pages_per_batch, dtype=torch.int32).reshape(batch, -1)
    cu_q = torch.arange(batch + 1, dtype=torch.int32) * q_len
    kv_lengths = torch.full((batch,), kv_len, dtype=torch.int32)

    out, lse, *_ = flash_attn_with_kvcache(
        q.npu(), k.npu(), v.npu(), cache_seqlens=kv_lengths.npu(),
        page_table=table.npu(), cu_seqlens_q=cu_q.npu(), max_seqlen_q=q_len,
        softmax_scale=1.0, num_splits=splits, return_softmax_lse=True,
    )
    expected_out = values.repeat_interleave(heads // kv_heads, dim=1)
    expected_out = expected_out[:, None].expand(batch, q_len, heads, dim).reshape_as(q)
    expected_lse = (scores + torch.log(torch.tensor(float(kv_len)))).T.contiguous()
    torch.testing.assert_close(out.cpu(), expected_out, atol=0, rtol=0)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-6, rtol=0)


@pytest.mark.parametrize("dtype,kv_len", [
    (torch.bfloat16, 2048),
    (torch.float16, 16384),
])
def test_fd_combine_keeps_partial_o_fp32(dtype, kv_len):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")

    # The two split means lie below their low-precision rounding midpoints.
    # Rounding either partial before combining changes the final output by
    # one ULP. Uniform attention has an exact FP32 mean as an independent oracle.
    heads, dim, page_size = 8, 64, 128
    q = torch.zeros((1, heads, dim), dtype=dtype, device="npu")
    k = torch.zeros((kv_len // page_size, page_size, 1, dim), dtype=dtype, device="npu")
    v = torch.ones((kv_len, 1, dim), dtype=dtype)
    v[:3] = 2
    v[kv_len // 2:kv_len // 2 + 10] = 2
    expected = v.float().mean(dim=0).to(dtype).expand_as(q.cpu())
    table = torch.arange(kv_len // page_size, dtype=torch.int32, device="npu")[None]
    lengths = torch.tensor([kv_len], dtype=torch.int32, device="npu")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    out, lse, *_ = flash_attn_with_kvcache(
        q, k, v.reshape_as(k).npu(), cache_seqlens=lengths,
        page_table=table, cu_seqlens_q=cu_q, max_seqlen_q=1,
        softmax_scale=1.0, num_splits=2, return_softmax_lse=True,
    )
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        lse.cpu(), torch.full((heads, 1), torch.log(torch.tensor(float(kv_len)))),
        atol=2e-6, rtol=0,
    )
