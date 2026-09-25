// MSL kernels for the full decode step. Compiled at runtime together with
// MslKernels.Source — mostly threadgroup-only style (the paravirt GPU lacks
// simdgroup reduction; simdgroup 8x8 MMA and fp16/bf16 ALU do exist).
namespace Sdcb.HyMT2Sharp.Backends.Metal;

internal static class MslDecodeKernels
{
    public const string Source = @"
#include <metal_stdlib>
#include <metal_extended_vector>
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

// paged KV: logical position j -> physical row via block table.
// tab[logicalBlock] = physical block index; physical row = phys * (1<<lg) + (j % (1<<lg)).
inline uint kv_row(device const int* tab, uint j, uint lg) {
    return ((uint)tab[j >> lg] << lg) | (j & ((1u << lg) - 1u));
}

// decode attention: one threadgroup per q head, 128 threads.
// scores over [0, kvLen) via paged block table, softmax, weighted V sum — all bf16 KV.
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
    device const int* tab [[buffer(10)]],
    constant int& lg [[buffer(11)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int group = heads / kvHeads;
    int kvh = (int)g / group;
    device const float* qh = q + g * headDim;
    device const ushort* kb = K + kvh * headDim;

    threadgroup float sc[4096];
    for (int t = (int)tid; t < kvLen; t += 128) {
        device const ushort* kk = kb + (ulong)kv_row(tab, (uint)t, (uint)lg) * kvStride;
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
            acc += sc[t] * bf16_to_f32(vb[(ulong)kv_row(tab, (uint)t, (uint)lg) * kvStride + d]);
        oh[d] = acc / sm;
    }
}

// Split-K decode attention (flash-decoding): threadgroup g = head * S +
// chunk computes softmax statistics (m, l) and the unnormalized weighted-V
// partial for its contiguous chunk of positions, written to
// part[g] = {m, l, acc[headDim]}. attn_merge combines the S partials per
// head. Parallelizes the serial position scan that dominates attn_decode
// at long context.
// INVARIANT: sc[1024] caps the per-chunk length at 1024 — the host must
// keep ceil(kvLen/S) <= 1024 (raise S or fall back to attn_decode).
kernel void attn_split(
    device const float* q [[buffer(0)]],
    device const ushort* K [[buffer(1)]],
    device const ushort* V [[buffer(2)]],
    device float* part [[buffer(3)]],           // [heads*S][2 + headDim]
    constant int& heads [[buffer(4)]],
    constant int& kvHeads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& kvStride [[buffer(7)]],
    constant int& kvLen [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    device const int* tab [[buffer(10)]],
    constant int& lg [[buffer(11)]],
    constant int& S [[buffer(12)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int h = (int)g / S;
    int chunk = (int)g - h * S;
    int group = heads / kvHeads;
    int kvh = h / group;
    device const float* qh = q + h * headDim;
    device const ushort* kb = K + kvh * headDim;

    int cl = (kvLen + S - 1) / S;
    int t0 = chunk * cl;
    int t1 = min(kvLen, t0 + cl);
    int n = t1 - t0;
    device float* pg = part + (ulong)g * (headDim + 2);
    if (n <= 0) {
        if ((int)tid == 0) { pg[0] = -3.4e38f; pg[1] = 0.0f; }
        for (int d = (int)tid; d < headDim; d += 128) pg[2 + d] = 0.0f;
        return;
    }

    threadgroup float sc[1024];
    for (int t = (int)tid; t < n; t += 128) {
        device const ushort* kk = kb + (ulong)kv_row(tab, (uint)(t0 + t), (uint)lg) * kvStride;
        float acc = 0.0f;
        for (int d = 0; d < headDim; d++)
            acc += qh[d] * bf16_to_f32(kk[d]);
        sc[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float red[128];
    float mx = -3.4e38f;
    for (int t = (int)tid; t < n; t += 128) mx = max(mx, sc[t]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sm = 0.0f;
    for (int t = (int)tid; t < n; t += 128) {
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
    for (int d = (int)tid; d < headDim; d += 128) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++)
            acc += sc[t] * bf16_to_f32(vb[(ulong)kv_row(tab, (uint)(t0 + t), (uint)lg) * kvStride + d]);
        pg[2 + d] = acc;
    }
    if ((int)tid == 0) { pg[0] = mx; pg[1] = sm; }
}

// out[h][d] = sum_i exp(m_i - m) * acc_i[d] / sum_i exp(m_i - m) * l_i
kernel void attn_merge(
    device const float* part [[buffer(0)]],    // [heads*S][2 + headDim]
    device float* out [[buffer(1)]],
    constant int& heads [[buffer(2)]],
    constant int& headDim [[buffer(3)]],
    constant int& S [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    device const float* ph = part + (ulong)g * S * (headDim + 2);
    threadgroup float red[128];
    float mx = -3.4e38f;
    for (int i = (int)tid; i < S; i += 128) mx = max(mx, ph[(ulong)i * (headDim + 2)]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];

    float den = 0.0f;
    for (int i = (int)tid; i < S; i += 128)
        den += exp(ph[(ulong)i * (headDim + 2)] - mx) * ph[(ulong)i * (headDim + 2) + 1];
    red[tid] = den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    den = red[0];

    for (int d = (int)tid; d < headDim; d += 128) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++)
            acc += exp(ph[(ulong)i * (headDim + 2)] - mx) * ph[(ulong)i * (headDim + 2) + 2 + d];
        out[g * headDim + d] = acc / den;
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

        int g0 = blk * 16 + j * 8 + (lq >> 2);  // sf[] is staged by global group index gi
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

// ===================== M3: multi-token (prefill) =====================
// TILE_T token rows share every decoded weight — the whole point of GEMM:
// weight bytes are read once per TILE_T tokens instead of once per token.
constant int TILE_T = 8;



// multi-token embed: one threadgroup per token, 256 threads stride the row.
kernel void q4k_embed_rows(
    device const uchar* embd [[buffer(0)]],
    device const int* toks [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int row = toks[g];
    const device uchar* r = embd + (ulong)row * (hidden >> 8) * 144;
    for (int i = (int)tid; i < hidden; i += 256)
        y[(ulong)g * hidden + i] = dequant_q4k(r + (i >> 8) * 144, i & 255);
}

kernel void q6k_embed_rows(
    device const uchar* embd [[buffer(0)]],
    device const int* toks [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int row = toks[g];
    const device uchar* r = embd + (ulong)row * (hidden >> 8) * 210;
    for (int i = (int)tid; i < hidden; i += 256) {
        int blk = i >> 8, j = (i >> 7) & 1, wq = i & 127, l = wq & 31, sub = wq >> 5;
        const device uchar* bp = r + blk * 210;
        int ql = bp[j * 64 + l], qh = bp[128 + j * 32 + l];
        int q6 = sub == 0 ? (ql & 0xF) | ((qh & 3) << 4)
             : sub == 1 ? (bp[j * 64 + 32 + l] & 0xF) | (((qh >> 2) & 3) << 4)
             : sub == 2 ? (ql >> 4) | (((qh >> 4) & 3) << 4)
                        : (bp[j * 64 + 32 + l] >> 4) | (((qh >> 6) & 3) << 4);
        float d = float(*(const device half*)(bp + 208));
        float s = float(*(const device char*)(bp + 192 + ((i & 255) >> 4)));
        y[(ulong)g * hidden + i] = d * s * float(q6 - 32);
    }
}

// RoPE over T consecutive positions: gid -> (t, head, pair); pos = posBase + t.
kernel void rope_neox_multi(
    device float* x [[buffer(0)]],
    constant int& headDim [[buffer(1)]],
    constant int& ropeDim [[buffer(2)]],
    constant int& posBase [[buffer(3)]],
    constant float& base [[buffer(4)]],
    constant int& nHeads [[buffer(5)]],
    constant int& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    int hd = ropeDim >> 1;
    int t = (int)gid / (nHeads * hd);
    int rem = (int)gid % (nHeads * hd);
    int h = rem / hd;
    int i = rem % hd;
    float freq = 1.0f / pow(base, float(i) / float(hd));
    float a = float(posBase + t) * freq;
    float c = cos(a), s = sin(a);
    device float* row = x + ((ulong)t * nHeads + h) * headDim;
    float x0 = row[i], x1 = row[i + hd];
    row[i] = x0 * c - x1 * s;
    row[i + hd] = x0 * s + x1 * c;
}

// ---- prefill attention: 2 dispatches/layer instead of T ----
// scores[h][t][j] = scale · (q[t,h] · K[j, kvHead(h)]), j < posBase+t+1.
// grid = (T, heads); one threadgroup per (token, q head), 128 threads
// scanning j positions strided by 128. K is bf16 [pos][kvStride].
kernel void attn_scores(
    device const float* q [[buffer(0)]],       // [T][heads*headDim]
    device const ushort* K [[buffer(1)]],
    device float* scores [[buffer(2)]],        // [heads][T][kvLen]
    constant int& heads [[buffer(3)]],
    constant int& kvHeads [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant int& kvStride [[buffer(6)]],
    constant int& kvLen [[buffer(7)]],
    constant int& posBase [[buffer(8)]],
    constant int& T [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    device const int* tab [[buffer(11)]],
    constant int& lg [[buffer(12)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    int t = (int)g.x, h = (int)g.y, tid = (int)tp.x;
    int kvh = h * kvHeads / heads;
    int kvHere = posBase + t + 1;
    if (kvHere > kvLen) kvHere = kvLen;
    threadgroup float qs[128];  // host guards headDim <= 128
    for (int i = tid; i < headDim; i += 128)
        qs[i] = q[(ulong)t * heads * headDim + h * headDim + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort* kb = K + kvh * headDim;
    for (int j = tid; j < kvHere; j += 128) {
        device const ushort* krow = kb + (ulong)kv_row(tab, (uint)j, (uint)lg) * kvStride;
        float acc = 0.0f;
        for (int i = 0; i < headDim; i++)
            acc += qs[i] * bf16_to_f32(krow[i]);
        scores[((ulong)h * T + t) * kvLen + j] = acc * scale;
    }
}

// out[t,h,d] = softmax_j(scores[h][t][j]) · V[j, kvHead(h), d].
// grid = (T, heads); stages the score row in threadgroup (kvLen <= 4096).
kernel void attn_combine(
    device const float* scores [[buffer(0)]],
    device const ushort* V [[buffer(1)]],
    device float* out [[buffer(2)]],           // [T][heads*headDim]
    constant int& heads [[buffer(3)]],
    constant int& kvHeads [[buffer(4)]],
    constant int& headDim [[buffer(5)]],
    constant int& kvStride [[buffer(6)]],
    constant int& kvLen [[buffer(7)]],
    constant int& posBase [[buffer(8)]],
    constant int& T [[buffer(9)]],
    device const int* tab [[buffer(10)]],
    constant int& lg [[buffer(11)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    int t = (int)g.x, h = (int)g.y, tid = (int)tp.x;
    int kvh = h * kvHeads / heads;
    int kvHere = posBase + t + 1;
    if (kvHere > kvLen) kvHere = kvLen;
    threadgroup float sc[4096];
    for (int j = tid; j < kvHere; j += 128)
        sc[j] = scores[((ulong)h * T + t) * kvLen + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float red[128];
    float mx = -3.4e38f;
    for (int j = tid; j < kvHere; j += 128) mx = max(mx, sc[j]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sm = 0.0f;
    for (int j = tid; j < kvHere; j += 128) {
        float e = exp(sc[j] - mx);
        sc[j] = e;
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
    for (int d = tid; d < headDim; d += 128) {
        float acc = 0.0f;
        for (int j = 0; j < kvHere; j++)
            acc += sc[j] * bf16_to_f32(vb[(ulong)kv_row(tab, (uint)j, (uint)lg) * kvStride + d]);
        out[(ulong)t * heads * headDim + h * headDim + d] = acc / sm;
    }
}

// ---- prefill attention through the MMA GEMM path ----
// Scores and probs·V are ordinary GEMMs (scores[t][j] = q_t·k_j, outT[d][t] =
// Sum_j V[j][d]·P[t][j]); staging them through mma_gemm turns the O(T·kvLen)
// scalar dots into tensor-core ops. These three helpers only reshape/convert
// operands: K -> fp16 rows, V -> fp32 rows, softmax over the score rows into
// fp16 P. Requires T % 64 == 0 (host falls back to attn_scores/attn_combine
// otherwise). Unused when the GPU lacks simdgroup MMA.

// K16[kvh][j][d] = half(K[j][kvh*headDim+d]). Flat grid over half4 packs.
kernel void k2h(
    device const ushort* K [[buffer(0)]],
    device half* K16 [[buffer(1)]],
    device const int* tab [[buffer(2)]],
    constant int& lg [[buffer(3)]],
    constant int& kvStride [[buffer(4)]],
    constant int& kvLen [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& kvHeads [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int packs = headDim >> 2;
    int i4 = (int)gid % packs;
    int jkvh = (int)gid / packs;
    int j = jkvh / kvHeads, kvh = jkvh % kvHeads;
    if (j >= kvLen) return;
    device const ushort* src = K + (ulong)kv_row(tab, (uint)j, (uint)lg) * kvStride
        + kvh * headDim + i4 * 4;
    half4 o;
    for (int i = 0; i < 4; i++) o[i] = (half)bf16_to_f32(src[i]);
    *(device half4*)(K16 + ((ulong)(kvh * kvLen + j) * headDim + i4 * 4)) = o;
}

// V32[j][kvh*headDim+d] = float(V[j][kvh*headDim+d]), j in [0, kvPad) — the
// padding rows are zeroed so the P·V MMA can round in_dim up to a tile.
// Same flat half4-pack mapping as k2h (fp32 out → float4 stores).
kernel void v2f(
    device const ushort* V [[buffer(0)]],
    device float* V32 [[buffer(1)]],
    device const int* tab [[buffer(2)]],
    constant int& lg [[buffer(3)]],
    constant int& kvStride [[buffer(4)]],
    constant int& kvLen [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& kvHeads [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int packs = headDim >> 2;
    int i4 = (int)gid % packs;
    int jkvh = (int)gid / packs;
    int j = jkvh / kvHeads, kvh = jkvh % kvHeads;
    float4 o = float4(0.0f);
    if (j < kvLen) {
        device const ushort* src = V + (ulong)kv_row(tab, (uint)j, (uint)lg) * kvStride
            + kvh * headDim + i4 * 4;
        for (int i = 0; i < 4; i++) o[i] = bf16_to_f32(src[i]);
    }
    *(device float4*)(V32 + ((ulong)j * kvHeads * headDim + kvh * headDim + i4 * 4)) = o;
}

// P16[h][t][j] = softmax_j(scores[h][t][j] * scale) over j < kvHere(t), zero
// to kvPad. grid (T, heads), 128 threads; kvLen <= 4096 (same bound as
// attn_combine's threadgroup row).
kernel void softmaxP(
    device const float* scores [[buffer(0)]],   // [h][T][kvLen]
    device half* P [[buffer(1)]],               // [h][T][kvPad]
    constant int& T [[buffer(2)]],
    constant int& kvLen [[buffer(3)]],
    constant int& kvPad [[buffer(4)]],
    constant int& posBase [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    int t = (int)g.x, h = (int)g.y, tid = (int)tp.x;
    int kvHere = posBase + t + 1;
    if (kvHere > kvLen) kvHere = kvLen;
    threadgroup float sc[4096];
    threadgroup float red[128];
    device const float* srow = scores + ((ulong)h * T + t) * kvLen;
    for (int j = tid; j < kvHere; j += 128)
        sc[j] = srow[j] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -3.4e38f;
    for (int j = tid; j < kvHere; j += 128) mx = max(mx, sc[j]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    float sm = 0.0f;
    for (int j = tid; j < kvHere; j += 128) { sc[j] = exp(sc[j] - mx); sm += sc[j]; }
    red[tid] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rs = 1.0f / red[0];
    device half* prow = P + ((ulong)h * T + t) * kvPad;
    for (int j = tid; j < kvPad; j += 128)
        prow[j] = j < kvHere ? (half)(sc[j] * rs) : (half)0.0f;
}

// append T tokens' K/V (fp32, [T x kDim]) into paged bf16 caches (block table) starting at posBase.
kernel void kv_append_multi(
    device const float* kSrc [[buffer(0)]],
    device const float* vSrc [[buffer(1)]],
    device ushort* kDst [[buffer(2)]],
    device ushort* vDst [[buffer(3)]],
    constant int& kDim [[buffer(4)]],
    constant int& kvStride [[buffer(5)]],
    constant int& posBase [[buffer(6)]],
    device const int* tab [[buffer(7)]],
    constant int& lg [[buffer(8)]],
    constant int& nRows [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid / kDim;
    int e = (int)gid % kDim;
    if (t >= nRows) return;
    uint prow = kv_row(tab, (uint)(posBase + t), (uint)lg);
    kDst[prow * kvStride + e] = f32_to_bf16(kSrc[gid]);
    vDst[prow * kvStride + e] = f32_to_bf16(vSrc[gid]);
}

// copy the first nRows rows of one physical KV block to another (both K and V).
// Used when appends diverge mid-block over a store-pinned physical block:
// the pinned block keeps its old contents for future restores, so the live
// logical block gets a fresh phys with its front rows copied over.
kernel void kv_copy_block(
    device const ushort* srcK [[buffer(0)]],
    device const ushort* srcV [[buffer(1)]],
    device ushort* dstK [[buffer(2)]],
    device ushort* dstV [[buffer(3)]],
    constant int& kvStride [[buffer(4)]],
    constant int& srcPhys [[buffer(5)]],
    constant int& dstPhys [[buffer(6)]],
    constant int& nRows [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid / kvStride;
    int e = (int)gid % kvStride;
    if (t >= nRows) return;
    ulong so = ((ulong)srcPhys * 64 + (uint)t) * (uint)kvStride + (uint)e;
    ulong dto = ((ulong)dstPhys * 64 + (uint)t) * (uint)kvStride + (uint)e;
    dstK[dto] = srcK[so];
    dstV[dto] = srcV[so];
}
// ===================== M5: extra quant formats =====================
// All follow the fast4 shape: 4 output cols per threadgroup, 64 threads/col.

// ---- Q8_0: 34B per 32 values (fp16 d + 32 int8). Row = (in_dim/32)*34B.
kernel void q8_0_embed_row(
    device const uchar* embd [[buffer(0)]],
    device const int* tok [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= hidden) return;
    int row = *tok;
    const device uchar* bp = embd + (ulong)row * (hidden >> 5) * 34 + (ulong)(gid >> 5) * 34;
    float d = float(*(const device half*)bp);
    y[gid] = d * float(*(const device char*)(bp + 2 + (gid & 31)));
}

kernel void q8_0_embed_rows(
    device const uchar* embd [[buffer(0)]],
    device const int* toks [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& hidden [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int row = toks[g];
    const device uchar* r = embd + (ulong)row * (hidden >> 5) * 34;
    for (int i = (int)tid; i < hidden; i += 256) {
        const device uchar* bp = r + (ulong)(i >> 5) * 34;
        y[(ulong)g * hidden + i] =
            float(*(const device half*)bp) * float(*(const device char*)(bp + 2 + (i & 31)));
    }
}

kernel void q8_0_gemv_fast4(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int sub = (int)(tid >> 6), st = (int)(tid & 63u);
    int bpr = in_dim >> 5;
    int col = (int)g * 4 + sub;
    bool valid = col < out_dim;
    if (!valid) col = out_dim - 1;
    const device uchar* row = w + (ulong)col * bpr * 34;

    float acc = 0.0f;
    for (int b = st; b < bpr; b += 64) {
        const device uchar* bp = row + b * 34;
        float d = float(*(const device half*)bp);
        float s = 0.0f;
        for (int k = 0; k < 8; k++) {
            char4 q = *(const device packed_char4*)(bp + 2 + k * 4);
            s += dot(float4(q.x, q.y, q.z, q.w),
                     *(const device float4*)(x + b * 32 + k * 4));
        }
        acc += d * s;
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


// ---- Q2_0C: 130B per 512 values (fp16 d + 128B quants, 4 codes/byte).
// value = ((byte >> 2c) & 3) * 2 - 3.  Row = (in_dim/512)*130B.
// Threads stride 64-weight sub-blocks (16 bytes) so in_dim=512 keeps 8
// threads busy per col instead of 1.
kernel void q2c_gemv_fast4(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int sub = (int)(tid >> 6), st = (int)(tid & 63u);
    int col = (int)g * 4 + sub;
    bool valid = col < out_dim;
    if (!valid) col = out_dim - 1;
    const device uchar* row = w + (ulong)col * (in_dim >> 9) * 130;

    float acc = 0.0f;
    int nsub = in_dim >> 6;
    for (int si = st; si < nsub; si += 64) {
        int blk = si >> 3, sj = si & 7;
        const device uchar* bp = row + blk * 130;
        float d = float(*(const device half*)bp);
        int xb = blk * 512 + sj * 64;
        float ss = 0.0f;
        for (int k = 0; k < 4; k++) {
            uchar4 u = *(const device packed_uchar4*)(bp + 2 + sj * 16 + k * 4);
            for (int bi = 0; bi < 4; bi++) {
                int ub = u[bi];
                float4 wv = float4(
                    float(((ub      ) & 3) * 2 - 3), float(((ub >> 2) & 3) * 2 - 3),
                    float(((ub >> 4) & 3) * 2 - 3),  float(((ub >> 6) & 3) * 2 - 3));
                ss += dot(wv, *(const device float4*)(x + xb + k * 16 + bi * 4));
            }
        }
        acc += d * ss;
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


// ---- STQ1_0: 42B per 256 values (Qs[32] = 64x 4-bit slots, Sign[8], fp16 d).
// group g -> slot nibble + sign bit -> qpack = CB[(sign<<4)|slot] (four 2-bit
// ternary lanes, code-1).  Weight j: chunk=j>>6, gloc=j&15, lane=(j>>4)&3.
// Row = (in_dim/256)*42B.  Threads stride 64-weight chunks; inner loop handles
// 4 groups at once so x reads stay vectorized (4 x float4 per 16 weights).
constant uchar STQ_CB[32] = {
    0xA9,0x89,0x29,0x09, 0xA6,0x86,0x26,0x06,
    0x9A,0x92,0x1A,0x12, 0x6A,0x62,0x4A,0x42,
    0x01,0x21,0x81,0xA1, 0x04,0x24,0x84,0xA4,
    0x10,0x18,0x90,0x98, 0x40,0x48,0x60,0x68 };

inline uint stq_qpack(uint idx) { return STQ_CB[idx & 31u]; }

// Decode the four ternary lanes of group i of a chunk into wv.
inline void stq_group(int i, uchar4 qa, uchar4 qb, uchar2 sg, thread float* wv)
{
    int byte_ = i < 8 ? qa[i >> 1] : qb[(i >> 1) - 4];
    int slot = (byte_ >> ((i & 1) * 4)) & 0xF;
    int sign = (sg[i >> 3] >> (i & 7)) & 1;
    uint qp = stq_qpack((uint)(sign << 4) | (uint)slot);
    wv[0] = float(int(qp       & 3u) - 1);
    wv[1] = float(int((qp >> 2) & 3u) - 1);
    wv[2] = float(int((qp >> 4) & 3u) - 1);
    wv[3] = float(int((qp >> 6) & 3u) - 1);
}

kernel void stq_gemv_fast4(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int sub = (int)(tid >> 6), st = (int)(tid & 63u);
    int col = (int)g * 4 + sub;
    bool valid = col < out_dim;
    if (!valid) col = out_dim - 1;
    const device uchar* row = w + (ulong)col * (in_dim >> 8) * 42;

    float acc = 0.0f;
    int nsub = in_dim >> 6;
    for (int si = st; si < nsub; si += 64) {
        int blk = si >> 2, c = si & 3;
        const device uchar* bp = row + blk * 42;
        float d = float(*(const device half*)(bp + 40));
        uchar4 qa = *(const device packed_uchar4*)(bp + c * 8);
        uchar4 qb = *(const device packed_uchar4*)(bp + c * 8 + 4);
        uchar2 sg = *(const device packed_uchar2*)(bp + 32 + c * 2);
        int xb = blk * 256 + c * 64;
        float ss = 0.0f;
        for (int i4 = 0; i4 < 4; i4++) {
            float w[4][4];
            for (int gg = 0; gg < 4; gg++) stq_group(i4 * 4 + gg, qa, qb, sg, w[gg]);
            float4 x0 = *(const device float4*)(x + xb + i4 * 4);
            float4 x1 = *(const device float4*)(x + xb + i4 * 4 + 16);
            float4 x2 = *(const device float4*)(x + xb + i4 * 4 + 32);
            float4 x3 = *(const device float4*)(x + xb + i4 * 4 + 48);
            for (int gg = 0; gg < 4; gg++)
                ss += w[gg][0] * x0[gg] + w[gg][1] * x1[gg]
                    + w[gg][2] * x2[gg] + w[gg][3] * x3[gg];
        }
        acc += d * ss;
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



// ===================== M6: tiled sgemm — fp16 weight scratch + transposed X.
// Weights are dequantized once per prefill call into a shared fp16 scratch
// buffer, X is transposed to fp32 xT[k][t] (stride Tpad). One shared GEMM
// serves all quants. fp16 halves the weight traffic and lets the MMA path use
// simdgroup_half8x8 fragments (fp32 accumulate — 2x fp32-MMA rate on M GPUs).
// per thread 4 tokens x 4 cols in registers, 8 vector loads per 4-k step.

// 32x32 tiled transpose: xT[k*Tpad + t] = x[t*in_dim + k]
kernel void xtranspose(
    device const float* x [[buffer(0)]],
    device float* xT [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& Tpad [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    threadgroup float tile[32 * 33];
    int tid = (int)tp.x;
    int row = tid >> 3, kq = tid & 7;
    int t = (int)g.y * 32 + row;
    int k = (int)g.x * 32 + kq * 4;
    float4 xv = float4(0.0f);
    if (t < T && k + 3 < in_dim)
        xv = *(const device float4*)(x + (ulong)t * in_dim + k);
    else if (t < T)
        for (int j = 0; j < 4 && k + j < in_dim; j++) xv[j] = x[(ulong)t * in_dim + k + j];
    tile[row * 33 + kq * 4 + 0] = xv.x;
    tile[row * 33 + kq * 4 + 1] = xv.y;
    tile[row * 33 + kq * 4 + 2] = xv.z;
    tile[row * 33 + kq * 4 + 3] = xv.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int ko = (int)g.x * 32 + (tid >> 3);
    int tq = tid & 7;
    if (ko < in_dim) {
        float4 o = float4(tile[(tq * 4 + 0) * 33 + (tid >> 3)],
                          tile[(tq * 4 + 1) * 33 + (tid >> 3)],
                          tile[(tq * 4 + 2) * 33 + (tid >> 3)],
                          tile[(tq * 4 + 3) * 33 + (tid >> 3)]);
        *(device float4*)(xT + (ulong)ko * Tpad + (int)g.y * 32 + tq * 4) = o;
    }
}

// y[t][c] = dot(x[t][:], w[c][:]). Grid: ((out+63)/64, ceil(T/64)). 256 threads:
// (tid&15) -> 4 tokens, (tid>>4) -> 4 cols. Tail-safe on both dims.
// tBase/cBase offset the covered region — used for the ragged edges the
// mma_gemm tiles don't cover. Weights arrive as fp16, widened on load.
kernel void sgemm4x4(
    device const half* w16 [[buffer(0)]],
    device const float* xT [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& Tpad [[buffer(6)]],
    constant int& tBase [[buffer(7)]],
    constant int& cBase [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    int tid = (int)tp.x;
    int ti = tid & 15, ci = tid >> 4;
    int t = tBase + (int)g.y * 64 + 4 * ti;
    int c = cBase + (int)g.x * 64 + 4 * ci;

    float4 acc0 = float4(0), acc1 = float4(0), acc2 = float4(0), acc3 = float4(0);
    device const float* a = xT + t;
    int c0 = min(c + 0, out_dim - 1);
    int c1 = min(c + 1, out_dim - 1);
    int c2 = min(c + 2, out_dim - 1);
    int c3 = min(c + 3, out_dim - 1);
    device const half* b0 = w16 + (ulong)c0 * in_dim;
    device const half* b1 = w16 + (ulong)c1 * in_dim;
    device const half* b2 = w16 + (ulong)c2 * in_dim;
    device const half* b3 = w16 + (ulong)c3 * in_dim;
    int nk = in_dim >> 2;
    for (int kk = 0; kk < nk; kk++) {
        float4 a0 = *(const device float4*)(a + (ulong)(4 * kk + 0) * Tpad);
        float4 a1 = *(const device float4*)(a + (ulong)(4 * kk + 1) * Tpad);
        float4 a2 = *(const device float4*)(a + (ulong)(4 * kk + 2) * Tpad);
        float4 a3 = *(const device float4*)(a + (ulong)(4 * kk + 3) * Tpad);
        float4 hb0 = float4(*(const device half4*)(b0 + 4 * kk));
        float4 hb1 = float4(*(const device half4*)(b1 + 4 * kk));
        float4 hb2 = float4(*(const device half4*)(b2 + 4 * kk));
        float4 hb3 = float4(*(const device half4*)(b3 + 4 * kk));
        acc0 += a0 * hb0.x + a1 * hb0.y + a2 * hb0.z + a3 * hb0.w;
        acc1 += a0 * hb1.x + a1 * hb1.y + a2 * hb1.z + a3 * hb1.w;
        acc2 += a0 * hb2.x + a1 * hb2.y + a2 * hb2.z + a3 * hb2.w;
        acc3 += a0 * hb3.x + a1 * hb3.y + a2 * hb3.z + a3 * hb3.w;
    }
    for (int i = 0; i < 4; i++) {
        int tt = t + i;
        if (tt < T) {
            if (c + 3 < out_dim) {
                *(device float4*)(y + (ulong)tt * out_dim + c) = float4(acc0[i], acc1[i], acc2[i], acc3[i]);
            } else {
                if (c0 >= 0 && c + 0 < out_dim) y[(ulong)tt * out_dim + c + 0] = acc0[i];
                if (c + 1 < out_dim) y[(ulong)tt * out_dim + c + 1] = acc1[i];
                if (c + 2 < out_dim) y[(ulong)tt * out_dim + c + 2] = acc2[i];
                if (c + 3 < out_dim) y[(ulong)tt * out_dim + c + 3] = acc3[i];
            }
        }
    }
}

// GEMM via simdgroup MMA 8x8 fragments (works on paravirt too — measured
// ~2x the scalar-FMA path). tg tile 64t x 64c, 8 simdgroups; each simdgroup
// computes a 16t x 32c sub-tile (2x4 fragments). k-blocks of 16 are staged
// in threadgroup memory as fp16: tA[k][t] (converted from fp32 xT) and
// tB[c][k] (fp16 weights straight from the dequant scratch). half8x8 A/B
// fragments multiply-accumulate into float8x8 accumulators — fp16 MMA is
// 2x fp32 on M-series, and fp32 accumulation keeps precision identical to
// the fp32-input version. Requires in_dim % 16 == 0; the host only
// dispatches this over full 64x64 tiles — ragged edges go to sgemm4x4.
kernel void mma_gemm(
    device const half* w16 [[buffer(0)]],   // [c][k] fp16
    device const float* xT [[buffer(1)]],   // [k][Tpad] fp32
    device float* y [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& Tpad [[buffer(6)]],
    uint2 g [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    threadgroup half tA[16 * 64];    // [k][t] fp16 (converted at stage time)
    threadgroup half tB[64 * 17];    // [c][k] padded +1 to dodge bank conflicts
    int sgid = (int)tp.x >> 5, tid = (int)tp.x;
    int st = (int)(sgid & 3) * 16;          // 4 sub-tiles of 16t
    int sc = (int)(sgid >> 2) * 32;         // 2 sub-tiles of 32c
    int t0 = (int)g.y * 64 + st;
    int c0 = (int)g.x * 64 + sc;
    int tgT = (int)g.y * 64, tgC = (int)g.x * 64;

    simdgroup_float8x8 acc[2][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    // A stage: 16k x 64t = 256 half4, one per thread: k=tid>>4, t4=(tid&15)*4
    int aK = tid >> 4, aT = (tid & 15) * 4;
    // B stage: 64c x 16k = 256 half4, one per thread: c=tid>>2, k4=(tid&3)*4
    int bC = tid >> 2, bK = (tid & 3) * 4;
    device const float* aSrc = xT + tgT + aT;
    device const half* bSrc = w16 + (ulong)(tgC + bC) * in_dim;

    for (int kb = 0; kb < in_dim; kb += 16) {
        *(threadgroup half4*)(tA + aK * 64 + aT) =
            half4(*(const device float4*)(aSrc + (ulong)(kb + aK) * Tpad));
        *(threadgroup half4*)(tB + bC * 17 + bK) =
            *(const device half4*)(bSrc + kb + bK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 16; kk += 8) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + kk * 64 + st, 64, ulong2(0,0), true);
            simdgroup_load(a1, tA + kk * 64 + st + 8, 64, ulong2(0,0), true);
            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, tB + (sc + 0) * 17 + kk, 17, ulong2(0,0), true);
            simdgroup_load(b1, tB + (sc + 8) * 17 + kk, 17, ulong2(0,0), true);
            simdgroup_load(b2, tB + (sc + 16) * 17 + kk, 17, ulong2(0,0), true);
            simdgroup_load(b3, tB + (sc + 24) * 17 + kk, 17, ulong2(0,0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], y + (ulong)(t0 + i * 8) * out_dim + c0 + j * 8, (ulong)out_dim);
}

// ---- dequant -> fp16 rows [col][k]; one thread per (col, granule-group).
// fp16 scratch feeds the half-fragment MMA GEMM (half the memory traffic of
// fp32, fp16 relative error ~1e-3 is far below quant noise).
// q4k: thread = 64 vals (2 group-pairs). grid (in_dim/64, out_dim).
kernel void q4k_deq(
    device const uchar* w [[buffer(0)]],
    device half* w16 [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& out_dim [[buffer(3)]],
    uint2 gp [[thread_position_in_grid]])
{
    int col = (int)gp.y;
    int pair = (int)gp.x;
    if (col >= out_dim || pair * 64 >= in_dim) return;
    const device uchar* row = w + (ulong)col * (in_dim >> 8) * 144;
    int blk = pair >> 2;
    int pi = (pair & 3) << 1;
    const device uchar* bp = row + blk * 144;
    int sc, mn, sc2, mn2;
    get_scale_min_k4(pi, bp + 4, sc, mn);
    get_scale_min_k4(pi + 1, bp + 4, sc2, mn2);
    float d  = float(*reinterpret_cast<const device half*>(bp));
    float dm = float(*reinterpret_cast<const device half*>(bp + 2));
    float2 s0 = float2(d * float(sc), dm * float(mn));
    float2 s1 = float2(d * float(sc2), dm * float(mn2));
    const device uint* q = reinterpret_cast<const device uint*>(bp + 16 + (pair & 3) * 32);
    device half* o = w16 + (ulong)col * in_dim + blk * 256 + (pi << 5);
    for (int wrd = 0; wrd < 8; wrd++) {
        uint q4 = q[wrd];
        float4 wvA, wvB;
        for (int j = 0; j < 4; j++) {
            uint nib = (q4 >> (j * 8)) & 0xffu;
            wvA[j] = s0.x * float(nib & 15u) - s0.y;
            wvB[j] = s1.x * float(nib >> 4) - s1.y;
        }
        *(device half4*)(o + wrd * 4) = half4(wvA);
        *(device half4*)(o + 32 + wrd * 4) = half4(wvB);
    }
}

// q6k: thread = 16 vals (one unit). grid (in_dim/16, out_dim).
kernel void q6k_deq(
    device const uchar* w [[buffer(0)]],
    device half* w16 [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& out_dim [[buffer(3)]],
    uint2 gp [[thread_position_in_grid]])
{
    int col = (int)gp.y;
    int u = (int)gp.x;
    if (col >= out_dim || u * 16 >= in_dim) return;
    const device uchar* row = w + (ulong)col * (in_dim >> 8) * 210;
    int blk = u >> 4;
    int j = (u >> 3) & 1;
    int lq = u & 7;
    const device uchar* bp = row + blk * 210;
    ushort2 ua = *(const device packed_ushort2*)(bp + j * 64 + lq * 4);
    ushort2 ub = *(const device packed_ushort2*)(bp + j * 64 + 32 + lq * 4);
    ushort2 uh = *(const device packed_ushort2*)(bp + 128 + j * 32 + lq * 4);
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
    float d = float(*(const device half*)(bp + 208));
    int g0 = j * 8 + (lq >> 2);  // scale index is local to this 256-block (16 int8 scales at bp+192)
    float s0 = d * float(*(const device char*)(bp + 192 + g0));
    float s1 = d * float(*(const device char*)(bp + 192 + g0 + 2));
    float s2 = d * float(*(const device char*)(bp + 192 + g0 + 4));
    float s3 = d * float(*(const device char*)(bp + 192 + g0 + 6));
    device half* o = w16 + (ulong)col * in_dim + base + lq * 4;
    *(device half4*)(o)      = half4(s0 * (float4(q0) - 32.0f));
    *(device half4*)(o + 32) = half4(s1 * (float4(q1) - 32.0f));
    *(device half4*)(o + 64) = half4(s2 * (float4(q2) - 32.0f));
    *(device half4*)(o + 96) = half4(s3 * (float4(q3) - 32.0f));
}

// q8_0: thread = 32 vals (one block). grid (in_dim/32, out_dim).
kernel void q8_0_deq(
    device const uchar* w [[buffer(0)]],
    device half* w16 [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& out_dim [[buffer(3)]],
    uint2 gp [[thread_position_in_grid]])
{
    int col = (int)gp.y;
    int blk = (int)gp.x;
    if (col >= out_dim || blk * 32 >= in_dim) return;
    const device uchar* bp = w + (ulong)col * (in_dim >> 5) * 34 + blk * 34;
    float d = float(*(const device half*)bp);
    device half* o = w16 + (ulong)col * in_dim + blk * 32;
    for (int k = 0; k < 8; k++) {
        char4 qv = *(const device packed_char4*)(bp + 2 + k * 4);
        *(device half4*)(o + k * 4) = half4(d * float4(float(qv.x), float(qv.y), float(qv.z), float(qv.w)));
    }
}

// q2c: thread = 64 vals (one sub). grid (in_dim/64, out_dim).
kernel void q2c_deq(
    device const uchar* w [[buffer(0)]],
    device half* w16 [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& out_dim [[buffer(3)]],
    uint2 gp [[thread_position_in_grid]])
{
    int col = (int)gp.y;
    int si = (int)gp.x;
    if (col >= out_dim || si * 64 >= in_dim) return;
    int blk = si >> 3, sj = si & 7;
    const device uchar* bp = w + (ulong)col * (in_dim >> 9) * 130 + blk * 130;
    float d = float(*(const device half*)bp);
    device half* o = w16 + (ulong)col * in_dim + blk * 512 + sj * 64;
    for (int k = 0; k < 4; k++) {
        uchar4 u = *(const device packed_uchar4*)(bp + 2 + sj * 16 + k * 4);
        for (int bi = 0; bi < 4; bi++) {
            int ub = u[bi];
            *(device half4*)(o + k * 16 + bi * 4) = half4(d * float4(
                float(((ub      ) & 3) * 2 - 3), float(((ub >> 2) & 3) * 2 - 3),
                float(((ub >> 4) & 3) * 2 - 3),  float(((ub >> 6) & 3) * 2 - 3)));
        }
    }
}

// stq: thread = 64 vals (one chunk). grid (in_dim/64, out_dim).
// group i -> positions {i, i+16, i+32, i+48} within the chunk.
kernel void stq_deq(
    device const uchar* w [[buffer(0)]],
    device half* w16 [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& out_dim [[buffer(3)]],
    uint2 gp [[thread_position_in_grid]])
{
    int col = (int)gp.y;
    int si = (int)gp.x;
    if (col >= out_dim || si * 64 >= in_dim) return;
    int blk = si >> 2, c = si & 3;
    const device uchar* bp = w + (ulong)col * (in_dim >> 8) * 42 + blk * 42;
    float d = float(*(const device half*)(bp + 40));
    uchar4 qa = *(const device packed_uchar4*)(bp + c * 8);
    uchar4 qb = *(const device packed_uchar4*)(bp + c * 8 + 4);
    uchar2 sg = *(const device packed_uchar2*)(bp + 32 + c * 2);
    device half* o = w16 + (ulong)col * in_dim + blk * 256 + c * 64;
    float wv[16][4];
    for (int i = 0; i < 16; i++) stq_group(i, qa, qb, sg, wv[i]);
    for (int lane = 0; lane < 4; lane++)
        for (int i4 = 0; i4 < 4; i4++)
            *(device half4*)(o + lane * 16 + i4 * 4) = half4(d * float4(
                wv[i4 * 4 + 0][lane], wv[i4 * 4 + 1][lane],
                wv[i4 * 4 + 2][lane], wv[i4 * 4 + 3][lane]));
}


// ===================== M7: fused decode kernels =====================
// gemv_multi: up to 3 weight sections in ONE dispatch. Each 4-column
// subgroup resolves its section from the global column index and branches
// on that section's quant type — q/k/v or gate+up pairs with mixed quant
// types (e.g. q4k + q6k) fuse without a per-type kernel.
// types: 0=Q4_K 1=Q6_K 2=Q8_0 3=Q2_0C 4=STQ1_0.
// flags: bit0 accumulate (y = acc + dot), bit1 pair-silu
// (y0[i] = silu(dot0)·dot1), bits 2/3/4 = per-section bf16 output,
// bit4 = rmsnorm: the threadgroup computes rms(x)·nrm once (the four
// subgroups share it) and applies it inside the dots — removes the
// standalone rms dispatch. Only worth it for small col counts: the
// redundant normalize scales with ncols (each group re-reads all of x
// through L2), so the host only sets this for sections with few cols.
inline ulong mv_rowbytes(int t, int in_dim) {
    switch (t) {
        case 0: return (ulong)(in_dim >> 8) * 144;
        case 1: return (ulong)(in_dim >> 8) * 210;
        case 2: return (ulong)(in_dim >> 5) * 34;
        case 3: return (ulong)(in_dim >> 9) * 130;
        default: return (ulong)(in_dim >> 8) * 42;
    }
}

// Scales entries a section needs in the staged table: q4k -> in_dim/32
// groups, q6k -> in_dim/16, others stage nothing.
inline int mv_ng(int t, int in_dim) {
    return t == 0 ? in_dim >> 5 : (t == 1 ? in_dim >> 4 : 0);
}

// Stage per-group scales into the subgroup's sf slice (K-quants only).
// sf has 384 float2 per subgroup — threadgroup memory is the occupancy
// limiter on paravirt GPUs; pair mode packs sections A+B into the same
// slice (A at [0,ngA), B at [ngA, ngA+ngB)).
inline void mv_stage(int t, const device uchar* row, int st, int in_dim,
                     threadgroup float2* sf) {
    if (t == 0) {
        int ng = in_dim >> 5;
        for (int gi = st; gi < ng; gi += 64) {
            const device uchar* bp = row + (gi >> 3) * 144;
            int sc, mn;
            get_scale_min_k4(gi & 7, bp + 4, sc, mn);
            float d = float(*reinterpret_cast<const device half*>(bp));
            float dm = float(*reinterpret_cast<const device half*>(bp + 2));
            sf[gi] = float2(d * float(sc), dm * float(mn));
        }
    } else if (t == 1) {
        int ng = in_dim >> 4;
        for (int gi = st; gi < ng; gi += 64) {
            const device uchar* bp = row + (gi >> 4) * 210;
            float d = float(*(const device half*)(bp + 208));
            float s = float(*(const device char*)(bp + 192 + (gi & 15)));
            sf[gi] = float2(d * s, -32.0f * d * s);
        }
    }
}

// One thread's partial dot for a row — identical math to the *_gemv_fast4
// kernels, with sf pointing at the subgroup's section slice instead of
// a global index base. When nrm != nullptr the effective input is
// x[i]*nrm[i]*rms — the fused pre-GEMV rmsnorm (norm weight folded per
// element, the scalar rms applied once to the accumulator).
inline float mv_dot(int t, const device uchar* row, device const float* x,
                    device const float* nrm, float rms,
                    int st, int in_dim, threadgroup const float2* sf) {
    switch (t) {
        case 0: {  // Q4_K — uint4 loads: 4 words (32 elems) per iteration
            float acc = 0.0f;
            int nwords = in_dim >> 3;
            for (int wi4 = st * 4; wi4 < nwords; wi4 += 256) {
                uint4 q4 = *reinterpret_cast<const device uint4*>(
                    row + ((wi4 >> 5) * 144) + 16 + ((wi4 & 31) >> 3) * 32 + ((wi4 & 7) << 2));
                for (int w = 0; w < 4; w++) {
                    int wi = wi4 + w;
                    int blk = wi >> 5;
                    int wpos = wi & 31;
                    int pi = wpos >> 3;
                    int g0 = pi << 1;
                    int kbase = blk * 256 + (pi << 6) + ((wpos & 7) << 2);
                    float2 s0 = sf[(blk << 3) + g0];
                    float2 s1 = sf[(blk << 3) + g0 + 1];
                    uint qw = q4[w];
                    if (nrm) {
                        for (int j = 0; j < 4; j++) {
                            uint nib = (qw >> (j * 8)) & 0xffu;
                            acc += x[kbase + j] * nrm[kbase + j] * (s0.x * float(nib & 15u) - s0.y);
                            acc += x[kbase + 32 + j] * nrm[kbase + 32 + j] * (s1.x * float(nib >> 4) - s1.y);
                        }
                    } else {
                        for (int j = 0; j < 4; j++) {
                            uint nib = (qw >> (j * 8)) & 0xffu;
                            acc += x[kbase + j]      * (s0.x * float(nib & 15u) - s0.y);
                            acc += x[kbase + 32 + j] * (s1.x * float(nib >> 4) - s1.y);
                        }
                    }
                }
            }
            return acc * rms;
        }
        case 1: {  // Q6_K
            float acc = 0.0f;
            int numPairs = in_dim >> 5;   // 2 units = 32 elems per iteration
            for (int u2 = st; u2 < numPairs; u2 += 64) {
                int u = u2 * 2;
                int blk = u >> 4;
                int j = (u >> 3) & 1;
                int lq = u & 7;   // even — (lq, lq+1) bytes are contiguous
                const device uchar* bp = row + blk * 210;
                ushort4 ua4 = *(const device packed_ushort4*)(bp + j * 64 + lq * 4);
                ushort4 ub4 = *(const device packed_ushort4*)(bp + j * 64 + 32 + lq * 4);
                ushort4 uh4 = *(const device packed_ushort4*)(bp + 128 + j * 32 + lq * 4);
                ushort2 ua = ua4.xy, ub = ub4.xy, uh = uh4.xy;
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
                int g0 = blk * 16 + j * 8 + (lq >> 2);
                float2 s0 = sf[g0],     s1 = sf[g0 + 2];
                float2 s2 = sf[g0 + 4], s3 = sf[g0 + 6];
                float4 x0 = *(const device float4*)(x + base + lq * 4);
                float4 x1 = *(const device float4*)(x + base + 32 + lq * 4);
                float4 x2 = *(const device float4*)(x + base + 64 + lq * 4);
                float4 x3 = *(const device float4*)(x + base + 96 + lq * 4);
                if (nrm) {
                    x0 *= *(const device float4*)(nrm + base + lq * 4);
                    x1 *= *(const device float4*)(nrm + base + 32 + lq * 4);
                    x2 *= *(const device float4*)(nrm + base + 64 + lq * 4);
                    x3 *= *(const device float4*)(nrm + base + 96 + lq * 4);
                }
                acc += dot(x0, s0.x * float4(q0) + s0.y) + dot(x1, s1.x * float4(q1) + s1.y)
                     + dot(x2, s2.x * float4(q2) + s2.y) + dot(x3, s3.x * float4(q3) + s3.y);
                // second unit of the pair
                int lq1 = lq + 1;
                ua = ua4.zw; ub = ub4.zw; uh = uh4.zw;
                int4 la1 = int4(ua.x & 0xFF, ua.x >> 8, ua.y & 0xFF, ua.y >> 8);
                int4 lb1 = int4(ub.x & 0xFF, ub.x >> 8, ub.y & 0xFF, ub.y >> 8);
                int4 hb1 = int4(uh.x & 0xFF, uh.x >> 8, uh.y & 0xFF, uh.y >> 8);
                int4 qA = int4(la1.x & 0xF, la1.y & 0xF, la1.z & 0xF, la1.w & 0xF) |
                          (int4(hb1.x & 3, hb1.y & 3, hb1.z & 3, hb1.w & 3) << 4);
                int4 qB = int4(lb1.x & 0xF, lb1.y & 0xF, lb1.z & 0xF, lb1.w & 0xF) |
                          (int4((hb1.x >> 2) & 3, (hb1.y >> 2) & 3, (hb1.z >> 2) & 3, (hb1.w >> 2) & 3) << 4);
                int4 qC = int4((la1.x >> 4) & 0xF, (la1.y >> 4) & 0xF, (la1.z >> 4) & 0xF, (la1.w >> 4) & 0xF) |
                          (int4((hb1.x >> 4) & 3, (hb1.y >> 4) & 3, (hb1.z >> 4) & 3, (hb1.w >> 4) & 3) << 4);
                int4 qD = int4((lb1.x >> 4) & 0xF, (lb1.y >> 4) & 0xF, (lb1.z >> 4) & 0xF, (lb1.w >> 4) & 0xF) |
                          (int4((hb1.x >> 6) & 3, (hb1.y >> 6) & 3, (hb1.z >> 6) & 3, (hb1.w >> 6) & 3) << 4);
                int g1 = blk * 16 + j * 8 + (lq1 >> 2);
                float2 t0 = sf[g1],     t1 = sf[g1 + 2];
                float2 t2 = sf[g1 + 4], t3 = sf[g1 + 6];
                float4 y0 = *(const device float4*)(x + base + lq1 * 4);
                float4 y1 = *(const device float4*)(x + base + 32 + lq1 * 4);
                float4 y2 = *(const device float4*)(x + base + 64 + lq1 * 4);
                float4 y3 = *(const device float4*)(x + base + 96 + lq1 * 4);
                if (nrm) {
                    y0 *= *(const device float4*)(nrm + base + lq1 * 4);
                    y1 *= *(const device float4*)(nrm + base + 32 + lq1 * 4);
                    y2 *= *(const device float4*)(nrm + base + 64 + lq1 * 4);
                    y3 *= *(const device float4*)(nrm + base + 96 + lq1 * 4);
                }
                acc += dot(y0, t0.x * float4(qA) + t0.y) + dot(y1, t1.x * float4(qB) + t1.y)
                     + dot(y2, t2.x * float4(qC) + t2.y) + dot(y3, t3.x * float4(qD) + t3.y);
            }
            return acc * rms;
        }
        case 2: {  // Q8_0 — char4 pairs cover both 16B halves per iter
            float acc = 0.0f;
            int bpr = in_dim >> 5;
            for (int b = st; b < bpr; b += 64) {
                const device uchar* bp = row + b * 34;
                float d = float(*(const device half*)bp);
                float s = 0.0f;
                for (int k = 0; k < 4; k++) {
                    int xb = b * 32 + k * 4;
                    char4 c0 = *(const device packed_char4*)(bp + 2 + k * 4);
                    char4 c1 = *(const device packed_char4*)(bp + 18 + k * 4);
                    float4 wv0 = float4(float(c0.x), float(c0.y), float(c0.z), float(c0.w));
                    float4 xv0 = *(const device float4*)(x + xb);
                    float4 wv1 = float4(float(c1.x), float(c1.y), float(c1.z), float(c1.w));
                    float4 xv1 = *(const device float4*)(x + xb + 16);
                    if (nrm) {
                        xv0 *= *(const device float4*)(nrm + xb);
                        xv1 *= *(const device float4*)(nrm + xb + 16);
                    }
                    s += dot(wv0, xv0) + dot(wv1, xv1);
                }
                acc += d * s;
            }
            return acc * rms;
        }
        case 3: {  // Q2_0C
            float acc = 0.0f;
            int nsub = in_dim >> 6;
            for (int si = st; si < nsub; si += 64) {
                int blk = si >> 3, sj = si & 7;
                const device uchar* bp = row + blk * 130;
                float d = float(*(const device half*)bp);
                int xb = blk * 512 + sj * 64;
                float ss = 0.0f;
                for (int k = 0; k < 4; k++) {
                    uchar4 u = *(const device packed_uchar4*)(bp + 2 + sj * 16 + k * 4);
                    for (int bi = 0; bi < 4; bi++) {
                        int ub = u[bi];
                        float4 wv = float4(
                            float(((ub      ) & 3) * 2 - 3), float(((ub >> 2) & 3) * 2 - 3),
                            float(((ub >> 4) & 3) * 2 - 3),  float(((ub >> 6) & 3) * 2 - 3));
                        float4 xv = *(const device float4*)(x + xb + k * 16 + bi * 4);
                        if (nrm) xv *= *(const device float4*)(nrm + xb + k * 16 + bi * 4);
                        ss += dot(wv, xv);
                    }
                }
                acc += d * ss;
            }
            return acc * rms;
        }
        default: {  // STQ1_0
            float acc = 0.0f;
            int nsub = in_dim >> 6;
            for (int si = st; si < nsub; si += 64) {
                int blk = si >> 2, c = si & 3;
                const device uchar* bp = row + blk * 42;
                float d = float(*(const device half*)(bp + 40));
                uchar4 qa = *(const device packed_uchar4*)(bp + c * 8);
                uchar4 qb = *(const device packed_uchar4*)(bp + c * 8 + 4);
                uchar2 sg = *(const device packed_uchar2*)(bp + 32 + c * 2);
                int xb = blk * 256 + c * 64;
                float ss = 0.0f;
                for (int i4 = 0; i4 < 4; i4++) {
                    float w[4][4];
                    for (int gg = 0; gg < 4; gg++) stq_group(i4 * 4 + gg, qa, qb, sg, w[gg]);
                    float4 x0 = *(const device float4*)(x + xb + i4 * 4);
                    float4 x1 = *(const device float4*)(x + xb + i4 * 4 + 16);
                    float4 x2 = *(const device float4*)(x + xb + i4 * 4 + 32);
                    float4 x3 = *(const device float4*)(x + xb + i4 * 4 + 48);
                    if (nrm) {
                        x0 *= *(const device float4*)(nrm + xb + i4 * 4);
                        x1 *= *(const device float4*)(nrm + xb + i4 * 4 + 16);
                        x2 *= *(const device float4*)(nrm + xb + i4 * 4 + 32);
                        x3 *= *(const device float4*)(nrm + xb + i4 * 4 + 48);
                    }
                    for (int gg = 0; gg < 4; gg++)
                        ss += w[gg][0] * x0[gg] + w[gg][1] * x1[gg]
                            + w[gg][2] * x2[gg] + w[gg][3] * x3[gg];
                }
                acc += d * ss;
            }
            return acc * rms;
        }
    }
}

kernel void gemv_multi(
    device const float* x [[buffer(0)]],
    device const uchar* w0 [[buffer(1)]],
    device const uchar* w1 [[buffer(2)]],
    device const uchar* w2 [[buffer(3)]],
    device uchar* y0 [[buffer(4)]],
    device uchar* y1 [[buffer(5)]],
    device uchar* y2 [[buffer(6)]],
    device const float* accIn [[buffer(7)]],
    constant int& in_dim [[buffer(8)]],
    constant int4& ncols [[buffer(9)]],
    constant int4& types [[buffer(10)]],
    constant int& flags [[buffer(11)]],
    device const float* nrmW [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    int sub = (int)(tid >> 6), st = (int)(tid & 63u);
    int c = (int)g * 4 + sub;
    bool pair = (flags & 2) != 0;
    // fused rmsnorm of x: all 256 threads of the group cooperate once — the
    // four subgroups share the result (per-subgroup redundancy would cost
    // 4x the L2 traffic for nothing).
    float rms = 1.0f;
    device const float* nrm = nullptr;
    threadgroup float rmsRed[256];
    if ((flags & 16) != 0) {
        float ss = 0.0f;
        for (int i = (int)tid; i < in_dim; i += 256) ss += x[i] * x[i];
        rmsRed[tid] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 128; s > 0; s >>= 1) {
            if ((int)tid < s) rmsRed[tid] += rmsRed[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        rms = rsqrt(rmsRed[0] / float(in_dim) + eps);
        nrm = nrmW;
    }
    const device uchar* rowA = w0;
    const device uchar* rowB = w1;
    device uchar* yp = y0;
    int tA = types.x, tB = types.y;
    int lc = c;
    bool valid = true;
    int bf = 0;
    if (pair) {
        valid = lc < ncols.x && ncols.x > 0;
        if (!valid) lc = max(ncols.x - 1, 0);
    } else {
        int s, l;
        if (c < ncols.x) { s = 0; l = c; }
        else if (c < ncols.x + ncols.y) { s = 1; l = c - ncols.x; }
        else { s = 2; l = c - ncols.x - ncols.y; }
        int sn = s == 0 ? ncols.x : (s == 1 ? ncols.y : ncols.z);
        valid = l >= 0 && l < sn;
        if (!valid) l = max(sn - 1, 0);
        tA = s == 0 ? types.x : (s == 1 ? types.y : types.z);
        rowA = s == 0 ? w0 : (s == 1 ? w1 : w2);
        yp = s == 0 ? y0 : (s == 1 ? y1 : y2);
        bf = (flags >> (2 + s)) & 1;
        lc = l;
    }
    const device uchar* rw = rowA + (ulong)lc * mv_rowbytes(tA, in_dim);
    threadgroup float2 sf[4 * 384];
    threadgroup float2* sfA = sf + sub * 384;
    threadgroup float2* sfB = sfA + mv_ng(tA, in_dim);
    mv_stage(tA, rw, st, in_dim, sfA);
    const device uchar* rwB = rowB;
    if (pair) {
        rwB = rowB + (ulong)lc * mv_rowbytes(tB, in_dim);
        mv_stage(tB, rwB, st, in_dim, sfB);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float accA = mv_dot(tA, rw, x, nrm, rms, st, in_dim, sfA);
    float accB = pair ? mv_dot(tB, rwB, x, nrm, rms, st, in_dim, sfB) : 0.0f;
    threadgroup float r1[256], r2[256];
    r1[tid] = accA;
    r2[tid] = accB;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s2 = 32; s2 > 0; s2 >>= 1) {
        if (st < s2) {
            r1[tid] += r1[tid + s2];
            if (pair) r2[tid] += r2[tid + s2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (st != 0 || !valid) return;
    if (pair) {
        float gv = r1[tid], uv = r2[tid];
        ((device float*)y0)[lc] = gv / (1.0f + exp(-gv)) * uv;
    } else if ((flags & 1) != 0) {
        ((device float*)yp)[lc] = accIn[lc] + r1[tid];
    } else if (bf != 0) {
        ((device ushort*)yp)[lc] = f32_to_bf16(r1[tid]);
    } else {
        ((device float*)yp)[lc] = r1[tid];
    }
}

// Fused NeoX-rope + per-head rmsnorm for q and k, one threadgroup per head
// row. q rows transform in place (fp32); k rows transform then land
// DIRECTLY in the bf16 KV row for `pos` — the separate kv_append hop
// disappears. headDim <= 256 (threadgroup staging).
kernel void rope_rms(
    device float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device ushort* kDst [[buffer(2)]],
    device const float* qNW [[buffer(3)]],
    device const float* kNW [[buffer(4)]],
    constant int& heads [[buffer(5)]],
    constant int& headDim [[buffer(6)]],
    constant int& ropeDim [[buffer(7)]],
    constant int& pos [[buffer(8)]],
    constant float& base [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    bool isQ = (int)g < heads;
    int r = isQ ? (int)g : (int)g - heads;
    threadgroup float rowv[256];
    threadgroup float red[256];
    if (isQ) {
        for (int i = (int)tid; i < headDim; i += 256) rowv[i] = q[(int)g * headDim + i];
    } else {
        for (int i = (int)tid; i < headDim; i += 256) rowv[i] = k[r * headDim + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int hd = ropeDim >> 1;
    if ((int)tid < hd) {
        float freq = 1.0f / pow(base, float(tid) / float(hd));
        float a = float(pos) * freq;
        float c = cos(a), s = sin(a);
        float x0 = rowv[tid], x1 = rowv[tid + hd];
        rowv[tid] = x0 * c - x1 * s;
        rowv[tid + hd] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc2 = 0.0f;
    for (int i = (int)tid; i < headDim; i += 256) acc2 += rowv[i] * rowv[i];
    red[tid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(red[0] / float(headDim) + eps);
    const device float* nw = isQ ? qNW : kNW;
    for (int i = (int)tid; i < headDim; i += 256) {
        float v = rowv[i] * scale * nw[i];
        if (isQ) q[(int)g * headDim + i] = v;
        else kDst[r * headDim + i] = f32_to_bf16(v);
    }
}

";
}
