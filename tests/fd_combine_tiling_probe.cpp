// Host-only FD schedule inspection for profiling; no device code is changed.
#include <sstream>
#include "../csrc/ascend950/flash_attn_npu_3/tiling.cpp"

static const char *Describe(const FAInferTilingData &t)
{
    static thread_local std::string result;
    std::ostringstream out;
    uint32_t aiv = t.fdActiveCoreNum * 2U;
    uint32_t rows = aiv ? std::min(8U, std::max(1U, (t.fdCombineRowNum + aiv - 1U) / aiv)) : 0U;
    out << "{\"flash_decode\":" << t.flashDecodeFlag
        << ",\"requested_splits\":" << t.requestedNumSplits
        << ",\"aic\":" << t.fdActiveCoreNum << ",\"aiv\":" << aiv
        << ",\"base_tasks\":" << t.fdBaseTaskNum
        << ",\"combine_rows\":" << t.fdCombineRowNum << ",\"row_tile\":" << rows
        << ",\"partial_tasks\":" << t.fdPartialTaskNum
        << ",\"row_capacity\":" << t.fdRowCapacity
        << ",\"lse_sub_stride\":" << t.fdLseSubStride
        << ",\"D\":" << t.embeddingSizeV << ",\"schedules\":[";
    for (uint32_t i = 0; i < t.fdCombineTaskNum; ++i) {
        if (i) out << ',';
        const auto &s = t.fdCombineSchedules[i];
        out << "{\"base\":" << s.baseTask << ",\"first_core\":" << s.firstCore
            << ",\"partial_start\":" << s.partialStart << ",\"partials\":" << s.partialCount << '}';
    }
    out << "]";
    out << ",\"combine_aivs\":" << t.fdCombineAivNum << ",\"task_info\":[";
    for (uint32_t i=0;i<t.fdCombineTaskNum;++i) {
        if(i) out << ',';
        auto &x=t.fdCombineTaskInfo[i];
        out << "[" << x.batchIdx << ',' << x.qStart << ',' << x.qRows << ',' << x.qHeadStart << ',' << x.rowNum << ']';
    }
    out << "],\"core_schedules\":[";
    for (uint32_t i=0;i<t.fdActiveCoreNum*2;++i) {
        if(i) out << ',';
        auto &x=t.fdCombineCoreSchedules[i];
        out << "[" << x.combineTaskIdx << ',' << x.sourceSubBlock << ',' << x.sourceLocalRow << ',' << x.rowCount << ']';
    }
    out << "],\"uniform_modes\":" << t.fdCombineUseUniform << "}";
    result = out.str();
    return result.c_str();
}

extern "C" size_t tiling_size() { return sizeof(FAInferTilingData); }
extern "C" const char *describe_tiling(const unsigned char *data)
{
    FAInferTilingData t{};
    std::memcpy(&t, data, sizeof(t));
    return Describe(t);
}

extern "C" void probe_raw(int batch, const int32_t *qLengths, int heads, int kvHeads, int dim,
                                    const int32_t *kvLengths, int cores, int splits, unsigned char *raw)
{
    std::vector<int32_t> cuQ(batch + 1);
    for (int i = 0; i <= batch; ++i) cuQ[i] = i ? cuQ[i-1] + qLengths[i-1] : 0;
    optiling::FAInferContext c{};
    c.numTokens = cuQ.back();
    c.numHeads = heads;
    c.embeddingSize = dim;
    c.embeddingSizeV = dim;
    c.blockSize = 128;
    c.kvHeads = kvHeads;
    c.batch = batch;
    c.maxQSeqlen = *std::max_element(qLengths, qLengths + batch);
    c.maxKvSeqlen = *std::max_element(kvLengths, kvLengths + batch);
    c.maxNumBlocksPerBatch = (c.maxKvSeqlen + 127) / 128;
    c.numBlocks = batch * c.maxNumBlocksPerBatch;
    c.qSeqlenList = cuQ.data();
    c.kvSeqlenList = kvLengths;
    c.scaleValue = 1.0F / std::sqrt(static_cast<float>(dim));
    c.maskType = optiling::MaskType::NO_MASK;
    c.dataType = optiling::DataType::BF16;
    c.pagedCacheFlag = true;
    c.isTnd = true;
    c.flashDecodeFlag = splits != 1 && c.maxQSeqlen <= 16 && c.maxKvSeqlen >= 1024;
    c.numSplits = splits;
    FAInferTilingData t{};
    optiling::FAInferTiling tiler(c);
    tiler.SetCoreNum(cores);
    tiler.DoTiling(t);
    std::memcpy(raw, &t, sizeof(t));
}
