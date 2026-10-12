# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("batch,q_len,heads,kv_heads,dim,splits,kv_len", [
    (1, 1, 64, 1, 128, 0, 2048),  # Decode target: parallel AIV row tiles.
    (2, 1, 8, 1, 128, 4, 2048),   # LSE head stride with multiple batches.
    (1, 3, 10, 2, 192, 3, 2048),  # Tail head blocks and tiles crossing heads.
    (2, 3, 6, 1, 59, 4, 2048),    # Packed partial O and TND output strides.
    (1, 16, 8, 1, 64, 4, 2048),   # Maximum merged Q-head row count.
    (1, 1, 1, 1, 1, 2, 2048),     # Only one source AIV has rows.
    (1, 1, 8, 1, 256, 2, 2048),   # Maximum supported head dimension.
    (2, 1, 5, 1, 59, 4, 2048),    # Uneven source AIV row ranges.
    (2, 3, 6, 1, 65, 4, 2048),    # Different input and output UB strides.
    (2, 1, 5, 1, 1, 4, 2048),     # Multi-row FP32 input stride of one block.
    (1, 16, 4, 1, 256, 0, 4096),   # Many auto splits span multiple UB load batches.
])
def test_fd_combine_output_and_lse(dtype, return_lse, batch, q_len, heads, kv_heads, dim, splits, kv_len):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")

    # Every key is identical, so attention is uniform. Different scores per
    # token/head and values per batch/KV head expose incorrect output scattering.
    total_q = batch * q_len
    q = torch.zeros((total_q, heads, dim), dtype=dtype)
    scores = torch.arange(total_q * heads).reshape(total_q, heads).float() / 128
    q[..., 0] = scores.to(dtype)
    page_size = 128
    pages_per_batch = kv_len // page_size
    k = torch.ones((batch * pages_per_batch, page_size, kv_heads, dim), dtype=dtype)
    values = (torch.arange(batch * kv_heads).reshape(batch, kv_heads, 1).float() + 1
              + (torch.arange(dim).float() % 4) / 4).to(dtype)
    v = values[:, None, None].expand(batch, pages_per_batch, page_size, kv_heads, dim)
    v = v.reshape_as(k).contiguous()
    table = torch.arange(batch * pages_per_batch, dtype=torch.int32).reshape(batch, -1)
    cu_q = torch.arange(batch + 1, dtype=torch.int32) * q_len
    kv_lengths = torch.full((batch,), kv_len, dtype=torch.int32)

    result = flash_attn_with_kvcache(
        q.npu(), k.npu(), v.npu(), cache_seqlens=kv_lengths.npu(),
        page_table=table.npu(), cu_seqlens_q=cu_q.npu(), max_seqlen_q=q_len,
        softmax_scale=1.0, num_splits=splits, return_softmax_lse=return_lse,
    )
    out = result[0] if return_lse else result
    expected_out = values.repeat_interleave(heads // kv_heads, dim=1)
    expected_out = expected_out[:, None].expand(batch, q_len, heads, dim).reshape_as(q)
    expected_lse = (scores + torch.log(torch.tensor(float(kv_len)))).T.contiguous()
    torch.testing.assert_close(out.cpu(), expected_out, atol=0, rtol=0)
    if return_lse:
        torch.testing.assert_close(result[1].cpu(), expected_lse, atol=2e-6, rtol=0)


@pytest.mark.parametrize("dtype,kv_len", [
    (torch.bfloat16, 2048),
    (torch.float16, 16384),
])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fd_combine_keeps_partial_o_fp32(dtype, kv_len, return_lse):
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

    result = flash_attn_with_kvcache(
        q, k, v.reshape_as(k).npu(), cache_seqlens=lengths,
        page_table=table, cu_seqlens_q=cu_q, max_seqlen_q=1,
        softmax_scale=1.0, num_splits=2, return_softmax_lse=return_lse,
    )
    out = result[0] if return_lse else result
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    if return_lse:
        torch.testing.assert_close(
            result[1].cpu(), torch.full((heads, 1), torch.log(torch.tensor(float(kv_len)))),
            atol=2e-6, rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("splits,logit_offset", [
    (3, 0.0),       # Uneven splits and padded reduction lanes.
    (16, 512.0),    # Large common offset must not enter Exp directly.
    (0, -512.0),    # Automatic splitting with large negative logits.
    (28, 128.0),    # Maximum physical split count on the target device.
])
def test_fd_combine_nonuniform_weights(dtype, return_lse, splits, logit_offset):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")

    heads, dim, kv_len, page_size = 8, 64, 8192, 128
    q = torch.zeros((1, heads, dim), dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros((kv_len, 1, dim), dtype=dtype)
    # Page-dependent scores give the partitions unequal softmax mass. Only
    # one Q/K component is nonzero, making the FP32 score oracle exact.
    logits = ((torch.arange(kv_len) // page_size) % 9 - 4).float() * 4
    k[:, 0, 0] = (logits + logit_offset).to(dtype)
    gen = torch.Generator().manual_seed(20261012)
    v = (torch.rand((kv_len, 1, dim), generator=gen) * 2 - 1).to(dtype)
    v[..., 0] = 1  # A constant channel also checks that the weights sum to one.
    scores = q[0].float() @ k[:, 0].float().T
    expected = (scores.softmax(dim=-1) @ v[:, 0].float()).unsqueeze(0)
    expected_lse = scores.logsumexp(dim=-1)[:, None]
    table = torch.arange(kv_len // page_size, dtype=torch.int32, device="npu")[None]
    lengths = torch.tensor([kv_len], dtype=torch.int32, device="npu")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    result = flash_attn_with_kvcache(
        q.npu(), k.reshape(-1, page_size, 1, dim).npu(),
        v.reshape(-1, page_size, 1, dim).npu(),
        cache_seqlens=lengths, page_table=table, cu_seqlens_q=cu_q,
        max_seqlen_q=1, softmax_scale=1.0, num_splits=splits,
        return_softmax_lse=return_lse,
    )
    out = (result[0] if return_lse else result).cpu()
    torch.testing.assert_close(
        out.float(), expected,
        atol=2e-4 if dtype == torch.bfloat16 else 3e-5,
        rtol=8e-3 if dtype == torch.bfloat16 else 2e-3,
    )
    torch.testing.assert_close(out[..., 0], torch.ones_like(out[..., 0]), atol=0, rtol=0)
    if return_lse:
        torch.testing.assert_close(result[1].cpu(), expected_lse, atol=1e-4, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fd_combine_weight_rows_and_reuse(dtype, return_lse):
    if "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("Ascend950 FD combine regression")

    batch, q_len, heads, dim, kv_len, page_size = 2, 3, 6, 65, 2048, 128
    lengths_cpu = [kv_len, kv_len - 2 * page_size]
    q = torch.zeros((batch * q_len, heads, dim), dtype=dtype)
    q[..., 0] = (torch.arange(batch * q_len * heads).reshape(-1, heads).float() / 16 - 1).to(dtype)
    k = torch.zeros((batch, kv_len, 1, dim), dtype=dtype)
    logits = ((torch.arange(kv_len) // page_size) % 7 - 3).float() * 2
    gen = torch.Generator().manual_seed(20261013)
    values = (torch.rand(k.shape, generator=gen) - 0.5).to(dtype)
    q_npu = q.npu()
    table = torch.arange(batch * kv_len // page_size, dtype=torch.int32, device="npu").reshape(batch, -1)
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device="npu")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="npu") * q_len

    # Different row scores and successive input changes expose stale VF
    # stores, incorrect row masks, and unintended cross-row broadcasting.
    for iteration in range(3):
        k[:, :, 0, 0] = (logits + iteration * 0.5).to(dtype)
        v = (values.float() * (-1 if iteration == 1 else 1) + iteration * 0.25).to(dtype)
        expected_out, expected_lse = [], []
        for bi, length in enumerate(lengths_cpu):
            query = q[bi * q_len:(bi + 1) * q_len].float().transpose(0, 1)
            scores = query @ k[bi, :length, 0].float().T
            expected_out.append((scores.softmax(dim=-1) @ v[bi, :length, 0].float()).transpose(0, 1))
            expected_lse.append(scores.logsumexp(dim=-1))
        result = flash_attn_with_kvcache(
            q_npu, k.reshape(-1, page_size, 1, dim).npu(),
            v.reshape(-1, page_size, 1, dim).npu(),
            cache_seqlens=lengths, page_table=table, cu_seqlens_q=cu_q,
            max_seqlen_q=q_len, softmax_scale=1.0, num_splits=3,
            return_softmax_lse=return_lse,
        )
        out = (result[0] if return_lse else result).cpu().float()
        torch.testing.assert_close(
            out, torch.cat(expected_out),
            atol=2e-4 if dtype == torch.bfloat16 else 3e-5,
            rtol=8e-3 if dtype == torch.bfloat16 else 2e-3,
        )
        if return_lse:
            torch.testing.assert_close(result[1].cpu(), torch.cat(expected_lse, dim=1), atol=1e-5, rtol=0)
