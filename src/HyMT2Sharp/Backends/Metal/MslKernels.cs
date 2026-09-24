// Metal Shading Language kernels for the Metal backend. Compiled at runtime via
// newLibraryWithSource:options:error: — no precompiled .metallib, no native toolchain.
namespace Sdcb.HyMT2Sharp.Backends.Metal;

internal static class MslKernels
{
    public const string Source = @"
#include <metal_stdlib>
using namespace metal;

inline void get_scale_min_k4(int index, const device uchar * packed, thread int & scale, thread int & minv) {
    if (index < 4) { scale = packed[index] & 63; minv = packed[index + 4] & 63; }
    else { scale = (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4);
           minv  = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4); }
}

inline float dequant_q4k(const device uchar * block, int within_block) {
    int group = within_block >> 5;
    int within_32 = within_block & 31;
    int pair_index = group >> 1;
    bool high_nibble = (group & 1) != 0;
    int sc = 0, mn = 0;
    get_scale_min_k4(group, block + 4, sc, mn);
    const half d_half = *reinterpret_cast<const device half *>(block);
    const half min_half = *reinterpret_cast<const device half *>(block + 2);
    const uchar packed_q = block[16 + pair_index * 32 + within_32];
    const int q = high_nibble ? ((packed_q >> 4) & 0x0f) : (packed_q & 0x0f);
    return float(d_half) * float(sc) * float(q) - float(min_half) * float(mn);
}

// portable: no simdgroup ops — threadgroup shared-mem tree reduction
kernel void q4k_gemv_portable(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int out_col = (int)g;
    if (out_col >= out_dim) return;
    int bpr = in_dim >> 8;
    threadgroup float red[256];
    float sum = 0.0f;
    const device uchar* row = w + (ulong)out_col * bpr * 144;
    for (int k = tid; k < in_dim; k += 256) {
        int blk = k >> 8, wb = k & 255;
        sum += x[k] * dequant_q4k(row + blk * 144, wb);
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[out_col] = red[0];
}

// v2: hoisted per-group scale decode + uint32 nibble loads, portable reduction
kernel void q4k_gemv_fast(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int out_col = (int)g;
    if (out_col >= out_dim) return;
    int bpr = in_dim >> 8;
    int ng = in_dim >> 5;
    const device uchar* row = w + (ulong)out_col * bpr * 144;

    threadgroup float2 sf[176];
    if ((int)tid < ng) {
        int blk = (int)tid >> 3;
        const device uchar* bp = row + blk * 144;
        int sc, mn;
        get_scale_min_k4((int)tid & 7, bp + 4, sc, mn);
        float d = float(*reinterpret_cast<const device half*>(bp));
        float dm = float(*reinterpret_cast<const device half*>(bp + 2));
        sf[tid] = float2(d * float(sc), dm * float(mn));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    int nwords = in_dim >> 3;
    for (int wi = (int)tid; wi < nwords; wi += 256) {
        int blk = wi >> 5;
        int wpos = wi & 31;
        int pi = wpos >> 3;
        int p4 = (wpos & 7) << 2;
        const device uchar* bp = row + blk * 144;
        uint q4 = *reinterpret_cast<const device uint*>(bp + 16 + pi * 32 + p4);
        int g0 = pi << 1;
        int kbase = blk * 256 + (pi << 6) + p4;
        float2 s0 = sf[(blk << 3) + g0], s1 = sf[(blk << 3) + g0 + 1];
        for (int j = 0; j < 4; j++) {
            uint nib = (q4 >> (j * 8)) & 0xffu;
            acc += x[kbase + j]      * (s0.x * float(nib & 15u) - s0.y);
            acc += x[kbase + 32 + j] * (s1.x * float(nib >> 4) - s1.y);
        }
    }
    threadgroup float red[256];
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[out_col] = red[0];
}

// v3: 4 output cols per threadgroup, 64 threads per col — amortize tg overhead
kernel void q4k_gemv_fast4(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int sub = (int)(tid >> 6);
    int st = (int)(tid & 63u);
    int bpr = in_dim >> 8;
    int ng = in_dim >> 5;
    int col = (int)g * 4 + sub;
    bool valid = col < out_dim;
    if (!valid) col = out_dim - 1;
    const device uchar* row = w + (ulong)col * bpr * 144;

    threadgroup float2 sf[4 * 176];
    for (int gi = st; gi < ng; gi += 64) {
        const device uchar* bp = row + (gi >> 3) * 144;
        int sc, mn;
        get_scale_min_k4(gi & 7, bp + 4, sc, mn);
        float d = float(*reinterpret_cast<const device half*>(bp));
        float dm = float(*reinterpret_cast<const device half*>(bp + 2));
        sf[sub * 176 + gi] = float2(d * float(sc), dm * float(mn));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    int nwords = in_dim >> 3;
    for (int wi = st; wi < nwords; wi += 64) {
        int blk = wi >> 5;
        int wpos = wi & 31;
        int pi = wpos >> 3;
        int p4 = (wpos & 7) << 2;
        const device uchar* bp = row + blk * 144;
        uint q4 = *reinterpret_cast<const device uint*>(bp + 16 + pi * 32 + p4);
        int g0 = pi << 1;
        int kbase = blk * 256 + (pi << 6) + p4;
        float2 s0 = sf[sub * 176 + (blk << 3) + g0];
        float2 s1 = sf[sub * 176 + (blk << 3) + g0 + 1];
        for (int j = 0; j < 4; j++) {
            uint nib = (q4 >> (j * 8)) & 0xffu;
            acc += x[kbase + j]      * (s0.x * float(nib & 15u) - s0.y);
            acc += x[kbase + 32 + j] * (s1.x * float(nib >> 4) - s1.y);
        }
    }
    threadgroup float red[256];
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 32; s > 0; s >>= 1) {
        if (st < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (st == 0 && valid) y[col] = red[tid];
}

// simdgroup fast path — may not run on paravirt GPU
kernel void q4k_gemv_simd(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int out_col = (int)g;
    if (out_col >= out_dim) return;
    int bpr = in_dim >> 8;
    uint lane = tid & 31u, sid = tid >> 5;
    threadgroup float partial[8];
    float sum = 0.0f;
    const device uchar* row = w + (ulong)out_col * bpr * 144;
    for (int k = tid; k < in_dim; k += 256) {
        int blk = k >> 8, wb = k & 255;
        sum += x[k] * dequant_q4k(row + blk * 144, wb);
    }
    float t = simd_sum(sum);
    if (lane == 0) partial[sid] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = lane < 8u ? partial[lane] : 0.0f;
    float f = simd_sum(v);
    if (tid == 0) y[out_col] = f;
}
";
}
