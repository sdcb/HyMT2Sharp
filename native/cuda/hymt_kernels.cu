// HyMT2Sharp-specific CUDA kernels. Compile to PTX with:
//   nvcc -ptx -arch=compute_80 -o native/ptx/hymt_kernels.ptx native/cuda/hymt_kernels.cu
// The PTX is committed so only nvcuda is needed at runtime (no toolkit).
#include <cuda_fp16.h>

// Dynamic-parameter slots, matching the vendored TensorSharp kernels' `dyn`
// buffer convention so one int[4] device buffer drives a whole CUDA-graph
// decode replay (written once per step via a captured HtoD memcpy node).
#define HYMT_DYN_ATTEND_LEN 0
#define HYMT_DYN_KV_WRITE_POS 1
#define HYMT_DYN_ROPE_POS 3

// Flat row-major rows [seq][heads*dim] -> head-first [heads][dstSeqStride][dim],
// with the seq rows written starting at position startPos. For seq == 1 this is
// a single-position KV append into a head-first cache (dstSeqStride = cache
// capacity); for seq > 1 it transposes activation rows into the head-first
// layout the GQA prefill/decode kernels read (dstSeqStride = seq for a
// contiguous temporary, cache capacity for a KV cache append).
extern "C" __global__ void hymt_rows_to_head_first_f32(
    const float* src,
    float* dst,
    int heads,
    int seq,
    int dim,
    int startPos,
    int dstSeqStride)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * dim;
    if (idx >= total)
        return;

    int d = (int)(idx % dim);
    long tmp = idx / dim;
    int s = (int)(tmp % seq);
    int h = (int)(tmp / seq);
    dst[((size_t)h * dstSeqStride + startPos + s) * dim + d] =
        src[((size_t)s * heads + h) * dim + d];
}

// Same transform, reading bf16 rows (the CPU engine's KV cache format) and
// writing f32. bf16 -> f32 is a left-shift of the bit pattern by 16.
extern "C" __global__ void hymt_rows_to_head_first_bf16(
    const unsigned short* src,
    float* dst,
    int heads,
    int seq,
    int dim,
    int startPos,
    int dstSeqStride)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * dim;
    if (idx >= total)
        return;

    int d = (int)(idx % dim);
    long tmp = idx / dim;
    int s = (int)(tmp % seq);
    int h = (int)(tmp / seq);
    unsigned int bits = ((unsigned int)src[((size_t)s * heads + h) * dim + d]) << 16;
    dst[((size_t)h * dstSeqStride + startPos + s) * dim + d] = __uint_as_float(bits);
}

// Same append as hymt_rows_to_head_first_f32 but reading the write position
// from the dyn buffer — needed so a captured CUDA-graph decode step can append
// KV at a per-replay position without re-instantiating.
extern "C" __global__ void hymt_rows_to_head_first_f32_dyn(
    const float* src,
    float* dst,
    int heads,
    int seq,
    int dim,
    const int* dyn,
    int dstSeqStride)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * dim;
    if (idx >= total || !dyn)
        return;

    int startPos = dyn[HYMT_DYN_KV_WRITE_POS];
    int d = (int)(idx % dim);
    long tmp = idx / dim;
    int s = (int)(tmp % seq);
    int h = (int)(tmp / seq);
    dst[((size_t)h * dstSeqStride + startPos + s) * dim + d] =
        src[((size_t)s * heads + h) * dim + d];
}

// h += b, then out = rmsnorm(h) * w — fuses the residual add with the RMSNorm
// that consumes the summed activations, so the norm is one node instead of two
// in every decode step. Single block: n stays small (hidden size).
extern "C" __global__ void hymt_add_rmsnorm_f32(
    float* h,
    const float* b,
    const float* w,
    float* out,
    int n,
    float eps)
{
    __shared__ float reduce[32];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float x = h[i] + b[i];
        h[i] = x;
        local += x * x;
    }
    // block reduce (sum) across warps
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, o);
    if ((threadIdx.x & 31) == 0)
        reduce[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        local = threadIdx.x < (blockDim.x + 31) / 32 ? reduce[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, o);
        if (threadIdx.x == 0)
            reduce[0] = local;
    }
    __syncthreads();
    float inv = rsqrtf(reduce[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = h[i] * inv * w[i];
}

// Fused per-head attention prep for decode: one launch replaces
// rope(Q)+rope(K)+per-head rmsnorm(Q)+rmsnorm(K)+append(K)+append(V).
// Grid = heads + 2*kvHeads blocks, blockDim >= dim (each block owns one head's
// dim-length row):
//   block b < heads            : Q head b — rope rotate (table row 0) then
//                                rmsnorm(qNormW), written back in place.
//   heads <= b < heads+kvHeads : K head (b-heads) — rope + rmsnorm(kNormW),
//                                written into kvK[b-heads][dyn[1]].
//   remaining blocks           : V head — copied into kvV[*][dyn[1]] (no norm).
extern "C" __global__ void hymt_attn_prep_f32(
    float* q,
    const float* k,
    const float* v,
    const float* qNormW,
    const float* kNormW,
    const float* cosTab,
    const float* sinTab,
    float* kvK,
    float* kvV,
    int heads,
    int kvHeads,
    int dim,
    int ropeHalf,
    int kvCap,
    float eps,
    const int* dyn)
{
    if (!dyn)
        return;
    int b = blockIdx.x;
    int pos = dyn[HYMT_DYN_KV_WRITE_POS];
    __shared__ float row[256];      // covers head_dim <= 256
    __shared__ float reduce[32];

    if (b < heads)
    {
        float* x = q + (size_t)b * dim;
        // NeoX rope on pairs (j, j+ropeHalf) using the single-position table.
        if (threadIdx.x < (unsigned)ropeHalf)
        {
            int j = threadIdx.x;
            float c = cosTab[j], s = sinTab[j];
            float x0 = x[j], x1 = x[j + ropeHalf];
            row[j] = x0 * c - x1 * s;
            row[j + ropeHalf] = x0 * s + x1 * c;
        }
        for (int d = threadIdx.x + ropeHalf * 2; d < dim; d += blockDim.x)
            row[d] = x[d];
        __syncthreads();
        float local = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            local += row[d] * row[d];
        for (int o = 16; o > 0; o >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, o);
        if ((threadIdx.x & 31) == 0)
            reduce[threadIdx.x >> 5] = local;
        __syncthreads();
        if (threadIdx.x < 32)
        {
            local = threadIdx.x < (blockDim.x + 31) / 32 ? reduce[threadIdx.x] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                local += __shfl_down_sync(0xffffffffu, local, o);
            if (threadIdx.x == 0)
                reduce[0] = local;
        }
        __syncthreads();
        float inv = rsqrtf(reduce[0] / dim + eps);
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            x[d] = row[d] * inv * qNormW[d];
        return;
    }

    int kh = b - heads;
    float* dst = (kh < kvHeads ? kvK : kvV)
        + ((size_t)(kh < kvHeads ? kh : kh - kvHeads) * kvCap + pos) * dim;
    const float* src = kh < kvHeads ? k + (size_t)kh * dim : v + (size_t)(kh - kvHeads) * dim;

    if (kh < kvHeads)
    {
        if (threadIdx.x < (unsigned)ropeHalf)
        {
            int j = threadIdx.x;
            float c = cosTab[j], s = sinTab[j];
            float x0 = src[j], x1 = src[j + ropeHalf];
            row[j] = x0 * c - x1 * s;
            row[j + ropeHalf] = x0 * s + x1 * c;
        }
        for (int d = threadIdx.x + ropeHalf * 2; d < dim; d += blockDim.x)
            row[d] = src[d];
        __syncthreads();
        float local = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            local += row[d] * row[d];
        for (int o = 16; o > 0; o >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, o);
        if ((threadIdx.x & 31) == 0)
            reduce[threadIdx.x >> 5] = local;
        __syncthreads();
        if (threadIdx.x < 32)
        {
            local = threadIdx.x < (blockDim.x + 31) / 32 ? reduce[threadIdx.x] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                local += __shfl_down_sync(0xffffffffu, local, o);
            if (threadIdx.x == 0)
                reduce[0] = local;
        }
        __syncthreads();
        float inv = rsqrtf(reduce[0] / dim + eps);
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            dst[d] = row[d] * inv * kNormW[d];
        return;
    }

    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        dst[d] = src[d];
}

// ---------------------------------------------------------------------------
// Decode matvec kernels (warp-per-row). The vendored vec kernels launch one
// CTA per output row with 4 threads cooperating on each 32-value sub-block;
// for a 1.8B model that is ~10-45us per matmul, ~4x off DRAM bandwidth.
// These kernels instead give each warp one output row and each lane one whole
// sub-block, mirroring ggml-cuda's mul_mat_vec shape: fewer CTAs, deeper ILP,
// and contiguous per-warp weight reads. Activation is the shared q8_1 row
// produced by ts_quantize_q8_1_* (same ts_block_q8_1 layout as the vendored
// kernels). HYMT2SHARP_CUDA_MATVEC=0 falls back to the vendored vec kernels.

__device__ __forceinline__ unsigned int hymt_read_u32_unaligned(const uint8_t* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ int hymt_dp4a(int a, int b, int c)
{
    return __dp4a(a, b, c);
}

__device__ __forceinline__ int hymt_get_scale_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index] & 0x3F;
    return (s[index + 4] & 0x0F) | ((s[index - 4] >> 6) << 4);
}

__device__ __forceinline__ int hymt_get_min_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index + 4] & 0x3F;
    return (s[index + 4] >> 4) | ((s[index] >> 6) << 4);
}

struct hymt_block_q8_1
{
    half d;
    half s;
    int8_t qs[32];
};

// Q4_K: one warp per output row, one lane per 32-value sub-block.
// Q4_K row: n_super * 144B (d f16, dmin f16, scales[12], qs[128]).
extern "C" __global__ void hymt_matvec_q4k_q81_f32(
    const uint8_t* weights,
    const hymt_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out_dim)
        return;

    int n_super = in_dim >> 8;   // 256 values per super-block
    int n_sub = in_dim >> 5;     // 32-value sub-blocks
    const uint8_t* w_row = weights + (size_t)row * n_super * 144;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int ib = lane; ib < n_sub; ib += 32)
    {
        int sb = ib >> 3;
        int ls = ib & 7;
        const uint8_t* sblock = w_row + (size_t)sb * 144;
        float d_sb = __half2float(*reinterpret_cast<const half*>(sblock));
        float dmin_sb = __half2float(*reinterpret_cast<const half*>(sblock + 2));
        const uint8_t* scales = sblock + 4;
        const uint8_t* w4 = sblock + 16 + (size_t)(ls >> 1) * 32;
        int shift = (ls & 1) * 4;
        const hymt_block_q8_1* ab = &xq[ib];

        int sumi = 0;
        // Q4_K rows are 144B so every row and sub-block stays 4B aligned;
        // the q8_1 qs field sits at offset 4 inside a 36B block, also 4B
        // aligned. Aligned u32 loads are safe here.
        const int* w4i = reinterpret_cast<const int*>(w4);
        const int* a4i = reinterpret_cast<const int*>(ab->qs);
        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            int w = (w4i[g] >> shift) & 0x0F0F0F0F;
            sumi = hymt_dp4a(w, a4i[g], sumi);
        }
        int sc = hymt_get_scale_k4(scales, ls);
        sumf_d += d_sb * (float)sc * __half2float(ab->d) * (float)sumi;
        int m = hymt_get_min_k4(scales, ls);
        sumf_m += dmin_sb * (float)m * __half2float(ab->s);
    }

    float acc = sumf_d - sumf_m;
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0)
        output[row] = acc;
}

// Q6_K: one warp per output row, one lane per 32-value sub-block.
// Q6_K row: n_super * 210B (ql[128], qh[64], scales[16] i8, d f16). A sub-block
// spans two 16-value scale groups, so the lane keeps two dp4a accumulators.
extern "C" __global__ void hymt_matvec_q6k_q81_f32(
    const uint8_t* weights,
    const hymt_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out_dim)
        return;

    int n_super = in_dim >> 8;
    int n_sub = in_dim >> 5;
    const uint8_t* w_row = weights + (size_t)row * n_super * 210;

    float acc = 0.0f;
    for (int ib = lane; ib < n_sub; ib += 32)
    {
        int sb = ib >> 3;
        int ls = ib & 7;
        const uint8_t* sblock = w_row + (size_t)sb * 210;
        const uint8_t* ql = sblock;
        const uint8_t* qh = sblock + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(sblock + 192);
        float d_sb = __half2float(*reinterpret_cast<const half*>(sblock + 208));

        int half_idx = ls >> 2;
        int group = ls & 3;
        const uint8_t* ql_group = ql + half_idx * 64 + ((group & 1) ? 32 : 0);
        const uint8_t* qh_group = qh + half_idx * 32;
        int ql_shift = group >= 2 ? 4 : 0;
        int qh_shift = group * 2;

        const hymt_block_q8_1* ab = &xq[ib];
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
        {
            int raw = ((hymt_read_u32_unaligned(ql_group + 4 * g) >> ql_shift) & 0x0F0F0F0F)
                    | (((hymt_read_u32_unaligned(qh_group + 4 * g) >> qh_shift) & 0x03030303) << 4);
            int w = __vsubss4(raw, 0x20202020);
            int sumi = hymt_dp4a(w, hymt_read_u32_unaligned((const uint8_t*)ab->qs + 4 * g), 0);
            if (g < 4) sumi0 += sumi; else sumi1 += sumi;
        }
        int sc0 = scales[half_idx * 8 + group * 2];
        int sc1 = scales[half_idx * 8 + group * 2 + 1];
        acc += d_sb * __half2float(ab->d) * ((float)sc0 * (float)sumi0 + (float)sc1 * (float)sumi1);
    }

    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0)
        output[row] = acc;
}
