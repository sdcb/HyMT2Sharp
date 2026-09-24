// MSL kernels for the full decode step. Compiled at runtime together with
// MslKernels.Source — same threadgroup-only style (no simdgroup assumptions,
// paravirt GPU lacks simdgroup reduction and bfloat ALU).
namespace Sdcb.HyMT2Sharp.Backends.Metal;

internal static class MslDecodeKernels
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

// Q6_K block: ql[128] @0, qh[64] @128, scales[16] i8 @192, d f16 @208 (210B/256 elems)
inline float dequant_q6k(const device uchar* block, int i) {
    int j = i >> 7, w = i & 127;
    int l = w & 31, sub = w >> 5;
    const device uchar* ql = block + j * 64;
    const device uchar* qh = block + 128 + j * 32;
    const device char* sc = (const device char*)(block + 192) + j * 8;
    int q6;
    if (sub == 0)      q6 = (ql[l] & 0xF)      | ((qh[l] & 3) << 4);
    else if (sub == 1) q6 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    else if (sub == 2) q6 = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4);
    else               q6 = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4);
    float d = float(*(const device half*)(block + 208));
    return d * float(sc[(l >> 4) + sub * 2]) * float(q6 - 32);
}

inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }
inline ushort f32_to_bf16(float f) {
    uint b = as_type<uint>(f);
    return ushort((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);  // RNE, matches Ops.ConvertToBf16
}

// embed: dequant one Q4_K row (token_embd[token]) into f32
kernel void q4k_embed_row(
    device const uchar* embd [[buffer(0)]],
    device const int* tok [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= hidden) return;
    int row = *tok;
    const device uchar* r = embd + (ulong)row * (hidden >> 8) * 144;
    int i = (int)gid;
    y[i] = dequant_q4k(r + (i >> 8) * 144, i & 255);
}

// rmsnorm over `dim` per row; one threadgroup per row, 256 threads.
// grid = rows; y[r] = x[r]·w / sqrt(mean(x[r]²)+eps)
kernel void rmsnorm_rows(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    device const float* xr = x + g * dim;
    float acc = 0.0f;
    for (int i = (int)tid; i < dim; i += 256) acc += xr[i] * xr[i];
    threadgroup float red[256];
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(red[0] / float(dim) + eps);
    device float* yr = y + g * dim;
    for (int i = (int)tid; i < dim; i += 256) yr[i] = xr[i] * scale * w[i];
}

// NeoX RoPE, in-place. grid = heads * (ropeDim/2); thread = (head, pair index)
kernel void rope_neox(
    device float* x [[buffer(0)]],
    constant int& headDim [[buffer(1)]],
    constant int& ropeDim [[buffer(2)]],
    constant int& pos [[buffer(3)]],
    constant float& base [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int hd = ropeDim >> 1;
    int h = (int)gid / hd;
    int i = (int)gid % hd;
    float freq = 1.0f / pow(base, float(i) / float(hd));
    float a = float(pos) * freq;
    float c = cos(a), s = sin(a);
    device float* row = x + h * headDim;
    float x0 = row[i], x1 = row[i + hd];
    row[i] = x0 * c - x1 * s;
    row[i + hd] = x0 * s + x1 * c;
}

// append one token's K and V (fp32) into flat bf16 caches at `pos`
kernel void kv_append_bf16(
    device const float* kSrc [[buffer(0)]],
    device const float* vSrc [[buffer(1)]],
    device ushort* kDst [[buffer(2)]],
    device ushort* vDst [[buffer(3)]],
    constant int& kDim [[buffer(4)]],
    constant int& kvStride [[buffer(5)]],
    constant int& pos [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= kDim) return;
    kDst[pos * kvStride + (int)gid] = f32_to_bf16(kSrc[gid]);
    vDst[pos * kvStride + (int)gid] = f32_to_bf16(vSrc[gid]);
}

// decode attention: one threadgroup per q head, 128 threads.
// scores over [0, kvLen), softmax, weighted V sum — all bf16 KV.
// kvLen <= 4096 (16KB threadgroup scores buffer).
kernel void attn_decode(
    device const float* q [[buffer(0)]],
    device const ushort* K [[buffer(1)]],
    device const ushort* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& heads [[buffer(4)]],
    constant int& kvHeads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& kvStride [[buffer(7)]],
    constant int& kvLen [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int group = heads / kvHeads;
    int kvh = (int)g / group;
    device const float* qh = q + g * headDim;
    device const ushort* kb = K + kvh * headDim;

    threadgroup float sc[4096];
    for (int t = (int)tid; t < kvLen; t += 128) {
        device const ushort* kk = kb + t * kvStride;
        float acc = 0.0f;
        for (int d = 0; d < headDim; d++)
            acc += qh[d] * bf16_to_f32(kk[d]);
        sc[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float red[128];
    float mx = -3.4e38f;
    for (int t = (int)tid; t < kvLen; t += 128) mx = max(mx, sc[t]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sm = 0.0f;
    for (int t = (int)tid; t < kvLen; t += 128) {
        float e = exp(sc[t] - mx);
        sc[t] = e;
        sm += e;
    }
    red[tid] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sm = red[0];

    device const ushort* vb = V + kvh * headDim;
    device float* oh = out + g * headDim;
    for (int d = (int)tid; d < headDim; d += 128) {
        float acc = 0.0f;
        for (int t = 0; t < kvLen; t++)
            acc += sc[t] * bf16_to_f32(vb[t * kvStride + d]);
        oh[d] = acc / sm;
    }
}

// gate[i] = silu(gate[i]) * up[i]
kernel void silu_mul(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    float gv = gate[gid];
    gate[gid] = gv / (1.0f + exp(-gv)) * up[gid];
}

// a[i] += b[i]
kernel void add_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    a[gid] += b[gid];
}

// debug: dequant one full row to float out (correctness bisect)
kernel void q6k_debug_row(
    device const uchar* w [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= in_dim) return;
    int i = (int)gid;
    const device uchar* block = w + (i >> 8) * 210;
    int j = (i >> 7) & 1, wq = i & 127, l = wq & 31, sub = wq >> 5;
    const device uchar* ql = block + j * 64;
    const device uchar* qh = block + 128 + j * 32;
    int q6;
    if (sub == 0)      q6 = (ql[l] & 0xF)      | ((qh[l] & 3) << 4);
    else if (sub == 1) q6 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    else if (sub == 2) q6 = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4);
    else               q6 = (ql[l + 32] >> 4)  | ((qh[l] >> 6) << 4);
    y[i] = float(q6);
}

// unit-based unpack debug: same dequant but through the uint32/int4 path
kernel void q6k_debug_row_u(
    device const uchar* w [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int u = (int)gid;                       // one unit per thread: 16 elems
    if (u >= (in_dim >> 4)) return;
    int blk = u >> 4, j = (u >> 3) & 1, lq = u & 7;
    const device uchar* bp = w + blk * 210;
    uchar4 la = *(const device packed_uchar4*)(bp + j * 64 + lq * 4);
    uchar4 lb = *(const device packed_uchar4*)(bp + j * 64 + 32 + lq * 4);
    uchar4 hb = *(const device packed_uchar4*)(bp + 128 + j * 32 + lq * 4);
    int base = blk * 256 + j * 128;
    int ba[4] = {la.x, la.y, la.z, la.w}, bb[4] = {lb.x, lb.y, lb.z, lb.w}, hf[4] = {hb.x, hb.y, hb.z, hb.w};
    for (int m = 0; m < 4; m++) {
        y[base +      lq*4+m] = (ba[m] & 0xF) | ((hf[m] & 3) << 4);
        y[base + 32 + lq*4+m] = (bb[m] & 0xF) | (((hf[m] >> 2) & 3) << 4);
        y[base + 64 + lq*4+m] = (ba[m] >> 4)  | (((hf[m] >> 4) & 3) << 4);
        y[base + 96 + lq*4+m] = (bb[m] >> 4)  | ((hf[m] >> 6) << 4);
    }
}

// embed for Q6_K token_embd (same contract as q4k_embed_row)
kernel void q6k_embed_row(
    device const uchar* embd [[buffer(0)]],
    device const int* tok [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= hidden) return;
    int row = *tok;
    const device uchar* r = embd + (ulong)row * (hidden >> 8) * 210;
    int i = (int)gid;
    y[i] = dequant_q6k(r + (i >> 8) * 210, i & 255);
}

// Q6_K gemv, same 4-col/64-thread structure as q4k_gemv_fast4.
// sf holds (d·sc, -32·d·sc) per 16-element group; in_dim <= 6144 → ng <= 384.
kernel void q6k_gemv_fast4(
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
    int ng = in_dim >> 4;
    int col = (int)g * 4 + sub;
    bool valid = col < out_dim;
    if (!valid) col = out_dim - 1;
    const device uchar* row = w + (ulong)col * bpr * 210;

    threadgroup float2 sf[4 * 384];
    for (int gi = st; gi < ng; gi += 64) {
        const device uchar* bp = row + (gi >> 4) * 210;
        float d = float(*(const device half*)(bp + 208));
        float s = float(*(const device char*)(bp + 192 + (gi & 15)));
        sf[sub * 384 + gi] = float2(d * s, -32.0f * d * s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each unit = 16 elements: one uint32 of ql low-nibble pairs + one of
    // high, one uint32 of qh — unpacked into int4 lanes.
    // u -> blk = u>>4, j = (u>>3)&1, lq = u&7 (l = lq*4); elems base+sub*32+l*4+m.
    float acc = 0.0f;
    int numUnits = in_dim >> 4;
    for (int u = st; u < numUnits; u += 64) {
        int blk = u >> 4;
        int j = (u >> 3) & 1;
        int lq = u & 7;
        const device uchar* bp = row + blk * 210;
        // 210B blocks are not 4B-aligned — packed_uchar4 (1B align) instead of uint
        // 210B rows misalign odd blocks to 2B — ushort2 (2B align) reads 4B in one op
        ushort2 ua = *(const device packed_ushort2*)(bp + j * 64 + lq * 4);      // ql[l..l+3] (sub0 low, sub2 high)
        ushort2 ub = *(const device packed_ushort2*)(bp + j * 64 + 32 + lq * 4); // ql[l+32..] (sub1 low, sub3 high)
        ushort2 uh = *(const device packed_ushort2*)(bp + 128 + j * 32 + lq * 4);// qh[l..l+3], 4x2-bit fields
        int4 la = int4(ua.x & 0xFF, ua.x >> 8, ua.y & 0xFF, ua.y >> 8);
        int4 lb = int4(ub.x & 0xFF, ub.x >> 8, ub.y & 0xFF, ub.y >> 8);
        int4 hb = int4(uh.x & 0xFF, uh.x >> 8, uh.y & 0xFF, uh.y >> 8);
        int base = blk * 256 + j * 128;

        int4 q0 = int4(la.x & 0xF, la.y & 0xF, la.z & 0xF, la.w & 0xF) |
                  (int4(hb.x & 3, hb.y & 3, hb.z & 3, hb.w & 3) << 4);
        int4 q1 = int4(lb.x & 0xF, lb.y & 0xF, lb.z & 0xF, lb.w & 0xF) |
                  (int4((hb.x >> 2) & 3, (hb.y >> 2) & 3, (hb.z >> 2) & 3, (hb.w >> 2) & 3) << 4);
        int4 q2 = int4((la.x >> 4) & 0xF, (la.y >> 4) & 0xF, (la.z >> 4) & 0xF, (la.w >> 4) & 0xF) |
                  (int4((hb.x >> 4) & 3, (hb.y >> 4) & 3, (hb.z >> 4) & 3, (hb.w >> 4) & 3) << 4);
        int4 q3 = int4((lb.x >> 4) & 0xF, (lb.y >> 4) & 0xF, (lb.z >> 4) & 0xF, (lb.w >> 4) & 0xF) |
                  (int4((hb.x >> 6) & 3, (hb.y >> 6) & 3, (hb.z >> 6) & 3, (hb.w >> 6) & 3) << 4);

        int g0 = (base + lq * 4) >> 4;
        float2 s0 = sf[sub * 384 + g0],     s1 = sf[sub * 384 + g0 + 2];
        float2 s2 = sf[sub * 384 + g0 + 4], s3 = sf[sub * 384 + g0 + 6];
        float4 x0 = *(const device float4*)(x + base + lq * 4);
        float4 x1 = *(const device float4*)(x + base + 32 + lq * 4);
        float4 x2 = *(const device float4*)(x + base + 64 + lq * 4);
        float4 x3 = *(const device float4*)(x + base + 96 + lq * 4);
        acc += dot(x0, s0.x * float4(q0) + s0.y) + dot(x1, s1.x * float4(q1) + s1.y)
             + dot(x2, s2.x * float4(q2) + s2.y) + dot(x3, s3.x * float4(q3) + s3.y);
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
";
}
