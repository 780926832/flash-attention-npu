// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0.
// Host-only coverage checks; build with g++ -std=c++17 -O2 and run the binary.
#include <cassert>
#include <iostream>
#include <random>

#include "../csrc/ascend950/flash_attn_npu_3/tiling.cpp"

static void CheckCoverage(const FAInferTilingData &t)
{
    assert(t.fdCombineAivNum <= t.fdActiveCoreNum * FD_SOURCE_SUB_BLOCK_NUM);
    std::vector<std::vector<uint32_t>> coverage;
    uint64_t expectedLoad = 0U;
    for (uint32_t task = 0; task < t.fdCombineTaskNum; ++task) {
        const uint32_t rows = t.fdCombineTaskInfo[task].rowNum;
        coverage.emplace_back(rows, 0U);
        expectedLoad += static_cast<uint64_t>(rows) * t.fdCombineSchedules[task].partialCount;
    }
    uint64_t assignedLoad = 0U;
    for (uint32_t core = 0; core < MAX_FD_COMBINE_AIV_NUM; ++core) {
        const auto &s = t.fdCombineCoreSchedules[core];
        if (core >= t.fdCombineAivNum) {
            assert(s.rowCount == 0U);
            continue;
        }
        assert(s.combineTaskIdx < t.fdCombineTaskNum);
        assert(s.sourceSubBlock < FD_SOURCE_SUB_BLOCK_NUM);
        assert(s.rowCount != 0U);
        const uint32_t rows = t.fdCombineTaskInfo[s.combineTaskIdx].rowNum;
        const uint32_t split = std::min(rows, ((rows + 7U) / 8U * 8U) / 2U);
        const uint32_t sourceRows = s.sourceSubBlock == 0U ? split : rows - split;
        assert(s.sourceLocalRow + s.rowCount <= sourceRows);
        const uint32_t start = (s.sourceSubBlock == 0U ? 0U : split) + s.sourceLocalRow;
        for (uint32_t row = start; row < start + s.rowCount; ++row) {
            assert(++coverage[s.combineTaskIdx][row] == 1U);
        }
        assignedLoad += static_cast<uint64_t>(s.rowCount) *
            t.fdCombineSchedules[s.combineTaskIdx].partialCount;
    }
    for (const auto &task : coverage) {
        for (auto count : task) {
            assert(count == 1U);
        }
    }
    assert(assignedLoad == expectedLoad);
}

static FAInferTilingData MakeTiling(const std::vector<int32_t> &lengthsQ,
    const std::vector<int32_t> &lengthsKv, uint32_t heads, uint32_t kvHeads,
    uint32_t dim, uint32_t splits, uint32_t cores)
{
    std::vector<int32_t> cuQ(1, 0);
    for (int32_t length : lengthsQ) {
        cuQ.push_back(cuQ.back() + length);
    }
    optiling::FAInferContext c{};
    c.numTokens = cuQ.back();
    c.numHeads = heads;
    c.kvHeads = kvHeads;
    c.embeddingSize = c.embeddingSizeV = dim;
    c.batch = lengthsQ.size();
    c.maxQSeqlen = *std::max_element(lengthsQ.begin(), lengthsQ.end());
    c.maxKvSeqlen = *std::max_element(lengthsKv.begin(), lengthsKv.end());
    c.blockSize = 128;
    c.maxNumBlocksPerBatch = (c.maxKvSeqlen + 127) / 128;
    c.numBlocks = c.batch * c.maxNumBlocksPerBatch;
    c.qSeqlenList = cuQ.data();
    c.kvSeqlenList = lengthsKv.data();
    c.maskType = optiling::MaskType::NO_MASK;
    c.dataType = optiling::DataType::BF16;
    c.pagedCacheFlag = c.isTnd = c.flashDecodeFlag = true;
    c.numSplits = splits;
    FAInferTilingData t{};
    optiling::FAInferTiling tiler(c);
    tiler.SetCoreNum(cores);
    tiler.DoTiling(t);
    return t;
}

int main()
{
    auto target = MakeTiling({14, 14, 14, 14}, {8192, 2048, 2048, 2048}, 8, 1, 128, 16, 28);
    assert(target.flashDecodeFlag == 1U && target.fdCombineAivNum == 52U);
    assert(target.fdCombineUseUniform == 0U);
    assert(target.fdCombineTaskNum == 4U);
    uint32_t counts[4] = {};
    for (uint32_t i = 0; i < target.fdCombineAivNum; ++i) {
        const auto &s = target.fdCombineCoreSchedules[i];
        assert(s.rowCount == (s.combineTaskIdx == 0U ? 4U : 14U));
        ++counts[s.combineTaskIdx];
    }
    assert(counts[0] == 28U && counts[1] == 8U && counts[2] == 8U && counts[3] == 8U);
    assert(target.fdCombineSchedules[0].partialCount == 16);
    for (uint32_t i = 1; i < 4; ++i) {
        assert(target.fdCombineSchedules[i].partialCount == 4);
    }
    CheckCoverage(target);
    auto single = MakeTiling({1}, {8192}, 64, 1, 128, 16, 28);
    assert(single.fdCombineUseUniform ==
        (FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE));
    auto small = MakeTiling({14, 14, 14, 14}, {8192, 2048, 2048, 2048}, 8, 1, 64, 16, 28);
    assert(small.fdCombineUseUniform ==
        (FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE));
    auto smallSingleRound = MakeTiling({7, 7, 7, 7}, {8192, 2048, 2048, 2048}, 8, 1, 128, 16, 28);
    assert(smallSingleRound.fdCombineUseUniform == FD_COMBINE_UNIFORM_NO_LSE);

    std::mt19937 rng(20261013);
    for (uint32_t test = 0; test < 20000; ++test) {
        FAInferTilingData t{};
        t.fdActiveCoreNum = 1U + rng() % MAX_FD_ACTIVE_CORE_NUM;
        t.fdCombineTaskNum = 1U + rng() % std::min(t.fdActiveCoreNum, MAX_FD_COMBINE_TASK_NUM);
        for (uint32_t i = 0; i < t.fdCombineTaskNum; ++i) {
            t.fdCombineTaskInfo[i].rowNum = 1U + rng() % 128U;
            t.fdCombineSchedules[i].partialCount = 2U + rng() % 31U;
        }
        optiling::FillFdCombineCoreSchedules(t);
        CheckCoverage(t);
    }

    uint32_t fdCases = 0U;
    for (uint32_t test = 0; test < 4000; ++test) {
        const uint32_t batch = 1U + rng() % 7U;
        std::vector<int32_t> q(batch), kv(batch);
        for (uint32_t b = 0; b < batch; ++b) {
            q[b] = rng() % 145U;
            kv[b] = (1U + rng() % 64U) * 128U;
        }
        const uint32_t kvHeads = 1U + rng() % 3U;
        const uint32_t splits[] = {0, 2, 3, 8, 16, 28};
        auto t = MakeTiling(q, kv, (1U + rng() % 8U) * kvHeads, kvHeads,
            1U + rng() % 256U, splits[rng() % 6U], 2U + rng() % 31U);
        if (t.flashDecodeFlag != 0U) {
            CheckCoverage(t);
            ++fdCases;
        }
    }
    assert(fdCases > 100U);
    std::cout << "PASS: target layout, 20000 partition coverage cases, "
              << fdCases << " valid FD tilings from 4000 inputs\n";
}
