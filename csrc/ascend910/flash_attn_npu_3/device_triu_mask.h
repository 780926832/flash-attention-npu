/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Build the 2048x2048 int8 triu(1) mask on the current device. The tensor is
// a local per-call value and no host/device copy or AICPU work is involved.
inline at::Tensor MakeDeviceTriuMask()
{
    const c10::DeviceIndex idx = c10_npu::getCurrentNPUStream().device_index();
    constexpr int64_t dim = 2048;
    at::Tensor mask =
        at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    return mask;
}

#endif
