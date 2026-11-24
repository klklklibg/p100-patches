// GP100 (sm_60) fp16 matrix-vector path at batch size 1 (Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, IQ4_NL).
//
// sm_60 has no __dp4a: the generic MMVQ emulates every int8 dot with four scalar multiplies and is
// instruction-issue bound at ~45% of HBM2 bandwidth. GP100 does have full-rate half2 FMA, so here:
//   - activations are converted once to fp16, pre-scaled per 32-block by a power of two so that
//     max|y| < 2^15 (no overflow possible), with the exact fp32 block sum kept for the min term;
//   - weights are unpacked with byte permutes straight into SUBNORMAL half bit patterns, i.e. the
//     value q * 2^-24, which is exact and needs no magic-number subtraction; HFMA2 on sm_60 handles
//     subnormal operands at full rate;
//   - products accumulate in fp16 over one sub-block (16 terms per lane, bounded below 1.0), then
//     are flushed to fp32 with the sub-block scale.
// The result is ~10x closer to an fp64 reference than the q8_1 path it replaces (activation
// quantization to 8 bits dominates that path's error).
//
// Device code is compiled for sm_60 only, so every other architecture's SASS is unchanged.

#include "mmvq-gp100.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

// Blocks per SM the IQ3_S multi-column kernel is compiled for. 2 is what the register allocator lands on by
// itself (~200 registers); 3-4 halves the register budget and may spill, but gives the memory latency more
// warps to hide behind. Override with -DGP100_IQ3S_NC_MINBLK=3 to compare.
#ifndef GP100_IQ3S_NC_MINBLK
#define GP100_IQ3S_NC_MINBLK 2
#endif



// -------------------------------------------------------------------------------------------
// GPU-busy stamps (GGML_CUDA_GP100_TS=1)
//
// The per-launch event windows in gp100_prof_* bracket a launch on the stream, so they include
// whatever gap the launch queue leaves before the kernel actually starts. They therefore cannot
// tell "the GPU is working" from "the GPU is waiting for the next command". Every CTA here stamps
// its own entry and exit with %globaltimer - a device-wide 1 ns counter that keeps running inside a
// CUDA graph replay, where host-side events cannot be recorded at all. The union of those [t0,t1]
// intervals is the time this matmul actually had a CTA resident on the GPU.
// -------------------------------------------------------------------------------------------
__device__ unsigned long long * gp100_ts_buf = nullptr;  // 2 u64 per CTA: {t0, t1}
__device__ unsigned int         gp100_ts_cap = 0;        // capacity in CTAs
__device__ unsigned int         gp100_ts_n   = 0;        // CTAs that tried to stamp (monotonic)

static __device__ __forceinline__ unsigned long long gp100_ts_now() {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return 0;
#endif
}

// Raw PRMT, default mode: selector bit 3 means sign-replicate. __byte_perm promises to ignore that bit,
// so the compiler masks every selector with 0x7777; the IQ4_XS lookup below never needs the mask.
static __device__ __forceinline__ uint32_t gp100_prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

static __device__ __forceinline__ float gp100_byte_f(const uint32_t w, const int sel) {
    // byte `sel` of w -> float, exact: place it in the mantissa of 2^23 and subtract 2^23
    return __int_as_float(__byte_perm(w, 0x4B000000u, 0x7650 | sel)) - 8388608.f;
}

// MUL_MAT_ID (MoE): blockIdx.y is the expert slot within a token and blockIdx.z the token, so one pair is
// (expert matrix, activation column, output column) and the four pointers the kernels read through are
// rebased here while the kernel bodies stay untouched. Two shapes occur in a MoE layer:
//   down projection: src1 is [K, n_slots, n_tokens] -> every slot has its own activation column;
//   gate/up:         src1 is [K, 1, n_tokens]       -> the slots share the token's one column (ybc).
// ids == nullptr on the plain MUL_MAT path, where blockIdx.y and .z are always 0.
struct gp100_pairs {
    const void *   W;
    const half *   y;
    const float2 * ys;
    const float *  s16;
    float *        dst;
};

static __device__ __forceinline__ gp100_pairs gp100_pair_rebase(const gp100_pairs p, const int32_t * __restrict__ ids,
        const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
    if (ids == nullptr) {
        return p;
    }
    const int64_t slot = blockIdx.y;
    const int64_t tok  = blockIdx.z;
    const int64_t q    = tok * n_slots + slot; // output column, in (slot-major, token-minor) order
    const int64_t yc   = ybc ? tok : q;
    // ids is a view of [n_expert, n_tokens], so its rows are nb[1] apart no matter how many slots are used
    const int64_t ei   = (int64_t) tok * stride_ids + slot;
    gp100_pairs r;
    r.W   = (const char *) p.W + (int64_t) ids[ei] * stride_xpert;
    r.y   = p.y   + yc * stride_y;
    r.ys  = p.ys  + yc * n32;
    r.s16 = p.s16 + yc * 2 * n32;
    r.dst = p.dst + q * stride_dst;
    return r;
}

// x (f32, K) -> y (f16, K) with y = x * 2^k per 32-block, max|y| < 2^15;
// ys[b] = {2^(24-k), sum of the block's x}: 2^24 undoes the subnormal weight encoding;
// s16[2b + h] = sum of the block's 16-element half h (Q6_K sub-blocks are 16 wide).
static __global__ void gp100_prep_act(const float * __restrict__ x, half * __restrict__ y, float2 * __restrict__ ys,
                                      float * __restrict__ s16, const int nblocks, const int nb32 = 1 << 30, const int64_t stride_col = 0) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    const int b = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int l = threadIdx.x % WARP_SIZE;
    if (b >= nblocks) {
        return;
    }
    // column b / nb32 of src1 (stride_col floats apart), written contiguously: y is [ncols][K]
    const float v = x[(int64_t) (b / nb32) * stride_col + (b % nb32) * 32 + l];
    float a = fabsf(v);
    float s = v;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
        s += __shfl_xor_sync(0xFFFFFFFF, s, o);
    }
    if (l % 16 == 0) {
        s16[2 * b + l / 16] = s;
    }
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, 16));
    s += __shfl_xor_sync(0xFFFFFFFF, s, 16);
    int e = 0;
    if (a > 0.0f) {
        frexpf(a, &e); // a in [2^(e-1), 2^e)
    }
    const int k = 15 - e;
    y[b * 32 + l] = __float2half_rn(ldexpf(v, k));
    if (l == 0) {
        ys[b] = make_float2(ldexpf(1.0f, 24 - k), s);
    }
#else
    GGML_UNUSED_VARS(x, y, ys, s16, nblocks, nb32, stride_col);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Warp-reduce the RT row sums and store. With GLU the kernel ran rows [0,R) of ffn_up and [R,2R) of ffn_gate
// for the same outputs, and stores swiglu = silu(gate) * up (same formula as the unfused GLU op).
template <int R, bool GLU>
static __device__ __forceinline__ void gp100_store(float * sum, float * __restrict__ dst, const int row0, const int nrows, const int lane) {
    constexpr int RT = GLU ? 2*R : R;
#pragma unroll
    for (int r = 0; r < RT; r++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum[r] += __shfl_xor_sync(0xFFFFFFFF, sum[r], o);
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = GLU ? ggml_cuda_op_silu_single(sum[R + r]) * sum[r] : sum[r];
        }
    }
}

// Same, but the block's four warps each summed a different slice of K, so their sums are combined through
// shared memory before the store. For short matrices one warp per row over the whole K is a latency chain:
// a 24-row ssm_alpha at K=5120 walks 160 weight blocks serially, and the generic kernel beat us there.
template <int R, bool GLU>
static __device__ __forceinline__ void gp100_store_ksplit(float * sum, float * __restrict__ dst, const int row0,
                                                          const int nrows, const int lane, const int warp) {
    constexpr int RT = GLU ? 2*R : R;
    __shared__ float red[4][RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum[r] += __shfl_xor_sync(0xFFFFFFFF, sum[r], o);
        }
        if (lane == 0) {
            red[warp][r] = sum[r];
        }
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = red[0][r] + red[1][r] + red[2][r] + red[3][r];
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = GLU ? ggml_cuda_op_silu_single(sum[R + r]) * sum[r] : sum[r];
        }
    }
}

// One warp computes R rows. Within a warp, 4 groups of 8 threads each take one super-block per
// iteration; thread t of a group owns sub-blocks 2*(t/2) and 2*(t/2)+1, positions 16*(t%2)..+15.
// Q4_K and Q5_K share the super-block header (d, dmin, 12 bytes of 6-bit scales/mins) and the low-nibble
// layout; Q5_K adds 32 bytes of high bits between the header and the nibbles.
template <ggml_type type> struct gp100_kq;
template <> struct gp100_kq<GGML_TYPE_Q4_K> { typedef block_q4_K block; static constexpr int qs_u4 = 1; static constexpr bool has_qh = false; };
template <> struct gp100_kq<GGML_TYPE_Q5_K> { typedef block_q5_K block; static constexpr int qs_u4 = 3; static constexpr bool has_qh = true;  };

template <ggml_type type, int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_kq(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R; // with GLU: R rows of ffn_up, then the same R rows of ffn_gate
    GGML_UNUSED(s16);
    typedef typename gp100_kq<type>::block block;
    constexpr int U4_PER_BLOCK = sizeof(block) / sizeof(uint4); // q4_K 9, q5_K 11
    static_assert(sizeof(block) % sizeof(uint4) == 0, "block must be a whole number of uint4");

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g  = lane / 8;
    const int t  = lane % 8;
    const int jj = t / 2;
    const int hf = t % 2;

    const uint4 * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const block * W = (const block *) ((GLU && r >= R) ? vG : vW);
        const int row = min(row0 + r % R, nrows - 1); // clamp: out-of-range rows are computed but not stored
        base[r] = (const uint4 *) (W + (int64_t) row * stride_row + g);
    }

    // per-thread 6-bit scale decode: sub-blocks j<4: sc = A&63, m = B&63;
    // j>=4: sc = (C&0xF)|((A>>6)<<4), m = (C>>4)|((B>>6)<<4)   (A,B,C = scales[0..3], [4..7], [8..11])
    const bool     hi_sc = jj >= 2;
    const uint32_t mA = hi_sc ? 0u : 0x3F3F3F3Fu;
    const uint32_t mC = hi_sc ? 0x0F0F0F0Fu : 0u;
    const uint32_t mS = hi_sc ? 0x30303030u : 0u;
    const uint32_t mH = hf ? 0u : ~0u; // the two halves of a sub-block share its min: add it once
    const int      bsel = 2 * (jj & 1);

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint4 hd[RT], qh[RT], qs[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint4 * p = base[r] + (sb - g) * U4_PER_BLOCK;
            hd[r] = __ldg(p);                                  // d, dmin, scales[12]
            qh[r] = gp100_kq<type>::has_qh ? __ldg(p + 1 + hf) : make_uint4(0, 0, 0, 0); // q5_K high bits, positions 16*hf..
            qs[r] = __ldg(p + gp100_kq<type>::qs_u4 + t);      // low nibbles of sub-blocks 2jj (lo) and 2jj+1 (hi)
        }
        const uint4 * yp = (const uint4 *) (y + sb * 256 + 64 * jj + 16 * hf);
        const uint4 yv[4] = {__ldg(yp), __ldg(yp + 1), __ldg(yp + 4), __ldg(yp + 5)};
        const float4 sc4 = __ldg((const float4 *) (ys + sb * 8 + 2 * jj)); // {rescale_lo, sum_lo, rescale_hi, sum_hi}
        const half2 * YL = (const half2 *) &yv[0];
        const half2 * YH = (const half2 *) &yv[2];

#pragma unroll
        for (int r = 0; r < RT; r++) {
            half2 al = __float2half2_rn(0.0f);
            half2 ah = al;
            const uint32_t qsw[4] = {qs[r].x, qs[r].y, qs[r].z, qs[r].w};
            const uint32_t qhw[4] = {qh[r].x, qh[r].y, qh[r].z, qh[r].w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t w   = qsw[k];
                const uint32_t q   = qhw[k] >> (2 * jj);
                uint32_t vlo = w & 0x0F0F0F0Fu;          // one value per byte
                uint32_t vhi = (w >> 4) & 0x0F0F0F0Fu;
                if constexpr (gp100_kq<type>::has_qh) {  // q5_K: 5th bit
                    vlo |= (q & 0x01010101u) << 4;
                    vhi |= (q & 0x02020202u) << 3;
                }
                const uint32_t l0 = __byte_perm(vlo, 0, 0x4140); // bytes 0,1 -> two subnormal halves
                const uint32_t l1 = __byte_perm(vlo, 0, 0x4342); // bytes 2,3
                const uint32_t h0 = __byte_perm(vhi, 0, 0x4140);
                const uint32_t h1 = __byte_perm(vhi, 0, 0x4342);
                al = __hfma2(*(const half2 *) &l0, YL[2*k + 0], al);
                al = __hfma2(*(const half2 *) &l1, YL[2*k + 1], al);
                ah = __hfma2(*(const half2 *) &h0, YH[2*k + 0], ah);
                ah = __hfma2(*(const half2 *) &h1, YH[2*k + 1], ah);
            }
            const uint32_t A = hd[r].y;
            const uint32_t B = hd[r].z;
            const uint32_t C = hd[r].w;
            const uint32_t scw = (A & mA) | ( C       & mC) | ((A >> 2) & mS);
            const uint32_t mw  = ((B & mA) | ((C >> 4) & mC) | ((B >> 2) & mS)) & mH;
            const float2   dm  = __half22float2(*(const half2 *) &hd[r].x);
            const float    sl  = __half2float(__hadd(__low2half(al), __high2half(al)));
            const float    sh  = __half2float(__hadd(__low2half(ah), __high2half(ah)));
            sum[r] += dm.x * fmaf(gp100_byte_f(scw, bsel) * sc4.x, sl, gp100_byte_f(scw, bsel + 1) * sc4.z * sh)
                    - dm.y * fmaf(gp100_byte_f(mw,  bsel),  sc4.y,     gp100_byte_f(mw,  bsel + 1) * sc4.w);
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q4_K/Q5_K with 2..8 columns (speculative-decoding verification, small batches). Same weight path as the
// single-column kernel; each unpacked half2 of weights is FMA'd against every column, so the unpack cost -
// the expensive part on GP100 - is paid once per weight, not once per column. Kept separate from gp100_mmvq_kq
// on purpose: that one's load schedule is tuned in situ (see the note there).
template <ggml_type type, int R, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_kq_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    // v2: R rows stay in flight like the single-column kernel; each row's 32 weights per thread are unpacked
    // ONCE into 16 half2 registers, then every column is FMA'd against them. Column activations are re-read
    // per row through the read-only cache (all rows and warps of a block read the same addresses), so the
    // register cost of a column is 16, not 16 x R.
    typedef typename gp100_kq<type>::block block;
    constexpr int U4 = sizeof(block) / sizeof(uint4);
    const int K = nsb * QK_K;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8, t = lane % 8, jj = t / 2, hf = t % 2;

    const uint4 * base[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        base[r] = (const uint4 *) ((const block *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row + g);
    }
    const bool     hiS = jj >= 2;
    const uint32_t mA = hiS ? 0u : 0x3F3F3F3Fu, mC = hiS ? 0x0F0F0F0Fu : 0u, mS = hiS ? 0x30303030u : 0u;
    const int      bsel = 2 * (jj & 1);
    const uint32_t mH = hf ? 0u : ~0u;

    float sum[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            sum[r][c] = 0.0f;
        }
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint4 hd[R], qh[R], qs[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint4 * p = base[r] + (sb - g) * U4;
            hd[r] = __ldg(p);
            qh[r] = gp100_kq<type>::has_qh ? __ldg(p + 1 + hf) : make_uint4(0, 0, 0, 0);
            qs[r] = __ldg(p + gp100_kq<type>::qs_u4 + t);
        }
        const int yoff = sb * 256 + 64 * jj + 16 * hf;
        const int soff = sb * 8 + 2 * jj;
#pragma unroll
        for (int r = 0; r < R; r++) {
            uint32_t wl[8], wh[8]; // this row's 32 weights as subnormal half2: lo sub-block, hi sub-block
            const uint32_t qsw[4] = {qs[r].x, qs[r].y, qs[r].z, qs[r].w};
            const uint32_t qhw[4] = {qh[r].x, qh[r].y, qh[r].z, qh[r].w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t w = qsw[k], q = qhw[k] >> (2 * jj);
                uint32_t vlo = w & 0x0F0F0F0Fu, vhi = (w >> 4) & 0x0F0F0F0Fu;
                if constexpr (gp100_kq<type>::has_qh) {
                    vlo |= (q & 0x01010101u) << 4;
                    vhi |= (q & 0x02020202u) << 3;
                }
                wl[2*k] = __byte_perm(vlo, 0, 0x4140); wl[2*k + 1] = __byte_perm(vlo, 0, 0x4342);
                wh[2*k] = __byte_perm(vhi, 0, 0x4140); wh[2*k + 1] = __byte_perm(vhi, 0, 0x4342);
            }
            const uint32_t A = hd[r].y, B = hd[r].z, C = hd[r].w;
            const uint32_t scw = (A & mA) | (C & mC) | ((A >> 2) & mS);
            const uint32_t mw  = ((B & mA) | ((C >> 4) & mC) | ((B >> 2) & mS)) & mH;
            const float2 dm  = __half22float2(*(const half2 *) &hd[r].x);
            const float  sc0 = dm.x * gp100_byte_f(scw, bsel), sc1 = dm.x * gp100_byte_f(scw, bsel + 1);
            const float  m0  = dm.y * gp100_byte_f(mw,  bsel), m1  = dm.y * gp100_byte_f(mw,  bsel + 1);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + yoff);
                const uint4 y0 = __ldg(yp), y1 = __ldg(yp + 1), y2 = __ldg(yp + 4), y3 = __ldg(yp + 5);
                const float4 s4 = __ldg((const float4 *) (ys + (int64_t) c * (K / 32) + soff));
                const half2 * YL0 = (const half2 *) &y0; const half2 * YL1 = (const half2 *) &y1;
                const half2 * YH0 = (const half2 *) &y2; const half2 * YH1 = (const half2 *) &y3;
                half2 al = __float2half2_rn(0.0f), ah = al;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    al = __hfma2(*(const half2 *) &wl[k],     YL0[k], al);
                    al = __hfma2(*(const half2 *) &wl[4 + k], YL1[k], al);
                    ah = __hfma2(*(const half2 *) &wh[k],     YH0[k], ah);
                    ah = __hfma2(*(const half2 *) &wh[4 + k], YH1[k], ah);
                }
                const float sl = __half2float(__hadd(__low2half(al), __high2half(al)));
                const float sh = __half2float(__hadd(__low2half(ah), __high2half(ah)));
                float acc = sum[r][c];
                acc = fmaf(sc0, sl * s4.x, acc);
                acc = fmaf(sc1, sh * s4.z, acc);
                acc = fmaf(-m0, s4.y, acc);
                sum[r][c] = fmaf(-m1, s4.w, acc);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = sum[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) c * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// IQ4_XS: 8 sub-blocks of 32 with 6-bit scales, values from the 16-entry non-linear kvalues_iq4nl table.
// Thread t of a group owns sub-block t: qs[16t..16t+15], low nibbles = weights 0..15, high = 16..31.
// The table is stored +128 (all entries 1..241, so they fit an unsigned byte and a subnormal half);
// the 128 * sum(x) it adds is removed exactly in fp32 with the prep kernel's block sum.
// A 136-byte block is only 8-byte aligned, so it is loaded as uint2.
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_iq4_xs(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R; // with GLU: R rows of ffn_up, then the same R rows of ffn_gate
    GGML_UNUSED(s16);
    constexpr int U2_PER_BLOCK = sizeof(block_iq4_xs) / sizeof(uint2); // 17
    static_assert(sizeof(block_iq4_xs) % sizeof(uint2) == 0, "iq4_xs block must be a whole number of uint2");
    // kvalues_iq4nl + 128, little-endian bytes
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    const uint2 * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const int row = min(row0 + r % R, nrows - 1);
        base[r] = (const uint2 *) ((const block_iq4_xs *) ((GLU && r >= R) ? vG : vW) + (int64_t) row * stride_row + g);
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint2 hd[RT], q0[RT], q1[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint2 * p = base[r] + (sb - g) * U2_PER_BLOCK;
            hd[r] = __ldg(p);             // d | scales_h << 16, scales_l[4]
            q0[r] = __ldg(p + 1 + 2 * t); // qs[16t .. 16t+7]
            q1[r] = __ldg(p + 2 + 2 * t); // qs[16t+8 .. 16t+15]
        }
        const uint4 * yp = (const uint4 *) (y + sb * 256 + 32 * t);
        const uint4 yv[4] = {__ldg(yp), __ldg(yp + 1), __ldg(yp + 2), __ldg(yp + 3)};
        const float2 sc = __ldg(ys + sb * 8 + t); // {rescale, sum}
        const half2 * YL = (const half2 *) &yv[0]; // weights 0..15 of the sub-block
        const half2 * YH = (const half2 *) &yv[2]; // weights 16..31

#pragma unroll
        for (int r = 0; r < RT; r++) {
            half2 acc = __float2half2_rn(0.0f);
            const uint32_t qw[4] = {q0[r].x, q0[r].y, q1[r].x, q1[r].y};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                // nibbles of qw: [w(4k) lo, w(16+4k) hi, w(4k+1), w(17+4k), w(4k+2), w(18+4k), w(4k+3), w(19+4k)]
                const uint32_t q  = qw[k];
                // low-table lookups use q as is (indices >= 8 come out sign-replicated and are discarded by
                // the select); high-table lookups use q ^ 0x88888888, so bit 3 is clear exactly where needed
                const uint32_t qx = q ^ 0x88888888u;
                const uint32_t S  = ((q >> 1) & 0x44444444u) | 0x32103210u; // pick upper table half where bit 3 is set
                const uint32_t v0 = gp100_prmt(gp100_prmt(T0, T1, q),       gp100_prmt(T2, T3, qx),       S);
                const uint32_t v1 = gp100_prmt(gp100_prmt(T0, T1, q >> 16), gp100_prmt(T2, T3, qx >> 16), S >> 16);
                const uint32_t a = __byte_perm(v0, 0, 0x4240); // w(4k),    w(4k+1)
                const uint32_t b = __byte_perm(v1, 0, 0x4240); // w(4k+2),  w(4k+3)
                const uint32_t c = __byte_perm(v0, 0, 0x4341); // w(16+4k), w(17+4k)
                const uint32_t d = __byte_perm(v1, 0, 0x4341); // w(18+4k), w(19+4k)
                acc = __hfma2(*(const half2 *) &a, YL[2*k + 0], acc);
                acc = __hfma2(*(const half2 *) &b, YL[2*k + 1], acc);
                acc = __hfma2(*(const half2 *) &c, YH[2*k + 0], acc);
                acc = __hfma2(*(const half2 *) &d, YH[2*k + 1], acc);
            }
            const int   ls = ((hd[r].y >> (4 * t)) & 0xF) | (((hd[r].x >> (16 + 2 * t)) & 3) << 4);
            const float dl = __half2float(__ushort_as_half((unsigned short) (hd[r].x & 0xFFFF))) * (float) (ls - 32);
            const float s  = __half2float(__hadd(__low2half(acc), __high2half(acc)));
            sum[r] += dl * fmaf(s, sc.x, -128.0f * sc.y);
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// IQ4_XS with 2..8 columns: the table lookup is done once per weight word and FMA'd against every column.
template <int R, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_iq4_xs_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    constexpr int U2 = sizeof(block_iq4_xs) / sizeof(uint2); // 17
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;
    const int K = nsb * QK_K;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    const uint2 * base[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        base[r] = (const uint2 *) ((const block_iq4_xs *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row + g);
    }
    float sum[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            sum[r][c] = 0.0f;
        }
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint2 hd[R], q0[R], q1[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint2 * p = base[r] + (sb - g) * U2;
            hd[r] = __ldg(p);
            q0[r] = __ldg(p + 1 + 2 * t);
            q1[r] = __ldg(p + 2 + 2 * t);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            uint32_t wv[16]; // this row's 32 weights (table value + 128) as subnormal half2, in y order
            const uint32_t qw[4] = {q0[r].x, q0[r].y, q1[r].x, q1[r].y};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t q  = qw[k];
                const uint32_t qx = q ^ 0x88888888u;
                const uint32_t S  = ((q >> 1) & 0x44444444u) | 0x32103210u;
                const uint32_t v0 = gp100_prmt(gp100_prmt(T0, T1, q),       gp100_prmt(T2, T3, qx),       S);
                const uint32_t v1 = gp100_prmt(gp100_prmt(T0, T1, q >> 16), gp100_prmt(T2, T3, qx >> 16), S >> 16);
                wv[2*k]         = __byte_perm(v0, 0, 0x4240); // w(4k),    w(4k+1)
                wv[2*k + 1]     = __byte_perm(v1, 0, 0x4240); // w(4k+2),  w(4k+3)
                wv[8 + 2*k]     = __byte_perm(v0, 0, 0x4341); // w(16+4k), w(17+4k)
                wv[8 + 2*k + 1] = __byte_perm(v1, 0, 0x4341);
            }
            const int   ls = ((hd[r].y >> (4 * t)) & 0xF) | (((hd[r].x >> (16 + 2 * t)) & 3) << 4);
            const float dl = __half2float(__ushort_as_half((unsigned short) (hd[r].x & 0xFFFF))) * (float) (ls - 32);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + sb * 256 + 32 * t);
                const uint4 yv[4] = {__ldg(yp), __ldg(yp + 1), __ldg(yp + 2), __ldg(yp + 3)};
                const float2 sc = __ldg(ys + (int64_t) c * (K / 32) + sb * 8 + t);
                const half2 * Y = (const half2 *) yv;
                half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    acc = __hfma2(*(const half2 *) &wv[i], Y[i], acc);
                }
                const float sv = __half2float(__hadd(__low2half(acc), __high2half(acc)));
                sum[r][c] += dl * fmaf(sv, sc.x, -128.0f * sc.y);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = sum[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) c * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// IQ3_S: 110-byte super-block (d, 64 qs, 8 qh, 32 signs, 4 bytes of packed 4-bit scales).
// 110 is not a multiple of uint2, so every field is 2-byte aligned at best: qs and signs have to be
// read as uint16 pairs (a uint2 load would fault on odd blocks). The 512-entry grid table must be
// staged in shared memory: read straight from global it measured 122 GB/s, staged 202.8 GB/s, both
// on P100 at 8192x8192 against 179 GB/s for the generic int8 path.
// Thread t of a group owns unit t of a block: qs[8t..8t+7], qh[t], signs[4t..4t+3], scales[t/2].
// y arrives pre-scaled by 2^k per 32-block (gp100_prep_act), so every block sum is multiplied by
// ys.x * 2^-24 before it is accumulated. Omitting that factor makes the kernel read the right weights
// with the wrong amplitude and was the reason this path produced noise instead of text.
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_iq3_s(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R; // with GLU: R rows of ffn_up, then the same R rows of ffn_gate
    GGML_UNUSED(s16);

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    __shared__ uint32_t s_grid[512];
    for (int i = threadIdx.x; i < 512; i += 128) {
        s_grid[i] = iq3s_grid[i];
    }
    __syncthreads();

    const block_iq3_s * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const block_iq3_s * W = (const block_iq3_s *) ((GLU && r >= R) ? vG : vW);
        const int row = min(row0 + r % R, nrows - 1);
        base[r] = W + (int64_t) row * stride_row + g;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nsb; sb += 4) {
        const half * yb = y + sb * QK_K + 32 * t;
        // y carries a per-32-block scale of 2^k (see gp100_prep_act): ys.x = 2^(24-k) undoes it after the
        // dot product, so every block sum needs a further 2^-24. Neither IQ3_S nor IQ3_XXS offsets its
        // weights by +128, so the block sum in ys.y goes unused.
        const float yscale = __ldg(ys + sb * 8 + t).x * (1.0f / 16777216.0f);
        // 32 weights = 64 bytes = 4 x uint4: pull the activations into registers once
        const uint4 yv0 = __ldg((const uint4 *)(yb +  0));
        const uint4 yv1 = __ldg((const uint4 *)(yb +  8));
        const uint4 yv2 = __ldg((const uint4 *)(yb + 16));
        const uint4 yv3 = __ldg((const uint4 *)(yb + 24));
        const uint32_t yw[16] = {yv0.x, yv0.y, yv0.z, yv0.w,
                                 yv1.x, yv1.y, yv1.z, yv1.w,
                                 yv2.x, yv2.y, yv2.z, yv2.w,
                                 yv3.x, yv3.y, yv3.z, yv3.w};
        uint16_t q[4][RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t * p = base[r][sb - g].qs + 8 * t;
            q[0][r] = *(const uint16_t *)(p + 0);
            q[1][r] = *(const uint16_t *)(p + 2);
            q[2][r] = *(const uint16_t *)(p + 4);
            q[3][r] = *(const uint16_t *)(p + 6);
        }
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const block_iq3_s * b = base[r] + (sb - g);
            const int       qh = b->qh[t];
            const uint8_t * sg = b->signs + 4 * t;
            const uint32_t lo = (uint32_t) q[0][r] | ((uint32_t) q[1][r] << 16);
            const uint32_t hi = (uint32_t) q[2][r] | ((uint32_t) q[3][r] << 16);

            float acc = 0.0f;
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const int      qbyte = (l < 4) ? ((lo >> (8*l)) & 0xFF) : ((hi >> (8*(l-4))) & 0xFF);
                const int      idx = qbyte | ((qh << (8 - l)) & 0x100);
                const uint32_t gv  = s_grid[idx];
                const uint8_t  sb8 = sg[l >> 1];
                // two 4-bit signs share a byte: even l takes the low nibble, odd l the high one
                const int      nib = (l & 1) ? (sb8 >> 4) : (sb8 & 0x0F);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    int v = (gv >> (8*k)) & 0xFF;
                    if ((nib >> k) & 1) v = -v;
                    const uint32_t ypair = (k < 2) ? yw[2*l] : yw[2*l + 1];
                    const uint32_t yh    = (k & 1) ? (ypair >> 16) : (ypair & 0xFFFF);
                    acc = fmaf((float) v, __half2float(__ushort_as_half((unsigned short) yh)), acc);
                }
            }
            const uint8_t sc  = b->scales[t >> 1];
            const int     sn  = (t & 1) ? (sc >> 4) : (sc & 0xF);
            sum[r] += __half2float(b->d) * (float)(1 + 2*sn) * yscale * acc;
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

#if 0 // PTQ1_0 (Prism ternary) is a custom quant type absent from this tree; excluded at port time
// The 256-entry PTQ1_0 decode table: one byte holds four 2-bit trit codes (0 -> 0, 1 -> +1, 2/3 -> -1),
// and an entry holds the four resulting values as pairs of half. Same layout as vecdotq.cuh's
// vec_dot_ptq1_lut_entry, kept local so this file does not pull in the whole vec_dot header.
//
// The values are 2^-10 rather than 1.0 (0x1400 / 0x9400 instead of 0x3C00 / 0xBC00). prep_act scales an
// activation 32-block so its largest magnitude lands in [2^14, 2^15), which means one term is up to 2^15:
// a 32-term half2 sum of unit weights would overflow fp16 (max 65504) several times over and force an fp32
// accumulator, and fp32 costs 2.5x the instructions here (two cvt per pair, then adds). At 2^-10 a term is
// <= 2^5, 32 of them sum to <= 1024, and fp16 represents every integer up to 2048 exactly -- so the whole
// block accumulates in half2 with no rounding, and the 2^10 comes back once in yscale.
static __device__ __forceinline__ uint2 gp100_ptq1_lut_entry(const int i) {
    unsigned h[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int c = (i >> (2 * k)) & 3;
        h[k] = c == 0 ? 0x0000u : (c == 1 ? 0x1400u : 0x9400u); // 0.0, +2^-10, -2^-10 as half bits
    }
    uint2 e;
    e.x = h[0] | (h[1] << 16);
    e.y = h[2] | (h[3] << 16);
    return e;
}

// PTQ1_0 (Prism ternary, type 143): 128 weights per block at two bits per trit, no per-block scale --
// eight fp16 scales are hoisted into the 16-byte header of an eight-block group (PTQ1_0_QS/D), so a block
// is 32 B of quants at a group-strided offset. There is no sign plane and no subnormal packing here: the
// 256-entry LUT already holds +-1.0 as half, so the activation is used exactly as prep_act left it and
// only the usual ys.x * 2^-24 factor is applied at the end.
//
// Warp layout differs from the IQ3 kernels because the block is half as wide. lane = 4*g + t with g in
// [0,8) selecting one of eight blocks in flight and t in [0,4) taking 32 of that block's 128 weights; 32
// weights are 8 bytes = 8 LUT lookups and land exactly on one activation 32-block, so ys is indexed per
// (block, t) instead of per lane group. A K = 5120 row is 40 blocks, and eight per round divides it into
// five with no idle lane -- that alignment is the entire reason this can beat the generic kernel, where
// the same 40 blocks over 32 lanes leave the second round 75% empty (mmvq.cu: nwarps drops to 1 for
// exactly this reason, and still only covers 40/64 lanes).
//
// Accumulation is fp32: activations come out of prep_act scaled up to 2^15, so a half2 sum of 32 terms
// would overflow fp16 well before it filled (the reason the generic path can use half2 is that it folds
// int8 activations, |term| <= 128).
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_ptq1_0(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R;
    GGML_UNUSED(s16);

    // entry stamp: taken before the lookup table is built, so the interval covers the whole residency
    unsigned int ts_slot = 0xFFFFFFFFu;
    if (threadIdx.x == 0 && gp100_ts_buf != nullptr) {
        ts_slot = atomicAdd(&gp100_ts_n, 1u);
        if (ts_slot < gp100_ts_cap) {
            gp100_ts_buf[2u * ts_slot] = gp100_ts_now();
        }
    }

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g    = lane / 4;
    const int t    = lane % 4;

    __shared__ uint2 s_lut[256];
    for (int i = threadIdx.x; i < 256; i += 128) {
        s_lut[i] = gp100_ptq1_lut_entry(i);
    }
    __syncthreads();

    const int nb = 2 * nsb; // 128-weight blocks per row (nsb counts 256-weight super-blocks)
    // stride_row counts blocks (nb[1] / type_size), and a PTQ1_0 block is 34 bytes on average: eight
    // 32-byte payloads plus the eight fp16 scales that head the group.
    constexpr int BLOCK_BYTES = PTQ1_0_GROUP_BYTES / PTQ1_0_GROUP_BLOCKS; // 34

    const char * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const char * W   = (const char *) ((GLU && r >= R) ? vG : vW);
        const int    row = min(row0 + r % R, nrows - 1);
        base[r] = W + (int64_t) row * stride_row * BLOCK_BYTES;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nb; sb += 8) {
        const size_t qoff   = PTQ1_0_QS(sb) + 8 * t;
        const half * yb     = y + sb * 128 + 32 * t;
        // ys.x = 2^(24-k) undoes prep_act; 1024 undoes the 2^-10 baked into the LUT values
        const float  yscale = __ldg(ys + sb * 4 + t).x * (1024.0f / 16777216.0f);

        // 32 activations = 64 B = 4 x uint4, read once and reused by every row this warp owns
        const uint4 a0 = __ldg((const uint4 *) (yb +  0));
        const uint4 a1 = __ldg((const uint4 *) (yb +  8));
        const uint4 a2 = __ldg((const uint4 *) (yb + 16));
        const uint4 a3 = __ldg((const uint4 *) (yb + 24));
        const uint32_t yw[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                                 a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};

        uint2 wq[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            wq[r] = *(const uint2 *) (base[r] + qoff);
        }

#pragma unroll
        for (int r = 0; r < RT; r++) {
            // Four accumulators so the hfma2 chain has no serial dependency: each takes two of the eight
            // bytes, i.e. four terms of at most 2^5, and the tree reduction at the end keeps the total
            // under 512, well inside fp16's exact-integer range.
            __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint32_t w = (j < 4) ? wq[r].x : wq[r].y;
                const uint2    e = s_lut[(w >> (8 * (j & 3))) & 0xFFu]; // 4 trits -> 4 half
                const __half2  wl = *(const __half2 *) &e.x;
                const __half2  wh = *(const __half2 *) &e.y;
                const __half2  yl = *(const __half2 *) &yw[2*j    ];
                const __half2  yh = *(const __half2 *) &yw[2*j + 1];
                if (j & 1) {
                    h1 = __hfma2(wl, yl, h1);
                    h3 = __hfma2(wh, yh, h3);
                } else {
                    h0 = __hfma2(wl, yl, h0);
                    h2 = __hfma2(wh, yh, h2);
                }
            }
            const __half2 t = __hadd2(__hadd2(h0, h2), __hadd2(h1, h3));
            const float   acc = __low2float(t) + __high2float(t);
            const float   d = __half2float(__ushort_as_half(*(const unsigned short *) (base[r] + PTQ1_0_D(sb))));
            sum[r] += d * yscale * acc;
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);

    if (ts_slot < gp100_ts_cap) {
        gp100_ts_buf[2u * ts_slot + 1u] = gp100_ts_now();
    }
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row,
                     ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}
#endif // end PTQ1_0 excluded block

// IQ3_S with 2..8 columns. Same decomposition as the single-column kernel, but the 32 weights a unit
// decodes are packed once into 16 subnormal half2 and FMA'd against every column's activations, so the
// 110-byte super-block is read (and decoded) once per unit rather than once per column. That is where the
// verify batch wins: the weight stream is 41x larger than the activation stream at 5120 rows.
// iq3s_grid holds 1..15 per byte (E8 lattice), so byte v shifted left 6 sits at v << 6 <= 960, inside the
// 1023-wide subnormal mantissa and exact; the half then reads (v << 6) * 2^-24, and the sign nibble only
// sets bit 15 of each half (a negative subnormal). The 2^6 factor is folded into dl.
template <int R, int NC>
static __global__ void __launch_bounds__(128, GP100_IQ3S_NC_MINBLK) gp100_mmvq_iq3_s_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    GGML_UNUSED(ys);
    const int K = nsb * QK_K;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    // The unpack used to be the bottleneck, not the memory: turning one grid entry into two half2 took
    // ~11 integer ops (mask, shift, mask, shift, or ...) for every half2 that then costs ONE FMA, and on
    // Pascal the integer ALU shares its issue slots with the FP16 pipe, so only 43% of the slots were FMA.
    // Pre-shifting the grid removes the four mask/shift pairs, and the four-bit sign nibble becomes a
    // 16-entry lookup, which leaves 2 ops per half2. Both tables are tiny (4 KB + 64 B) and stay in shared.
    __shared__ uint2    s_grid[512];
    __shared__ uint2    s_sign[16];
    for (int i = threadIdx.x; i < 512; i += 128) {
        const uint32_t g = iq3s_grid[i];
        s_grid[i] = make_uint2((((g      ) & 0xFFu) << 6) | (((g >>  8) & 0xFFu) << 22),
                               (((g >> 16) & 0xFFu) << 6) | (((g >> 24) & 0xFFu) << 22));
    }
    if (threadIdx.x < 16) {
        // .x signs the low half2 of a grid entry (nibble bits 0 and 1), .y the high one (bits 2 and 3)
        s_sign[threadIdx.x] = make_uint2(((threadIdx.x & 1u) << 15) | ((threadIdx.x & 2u) << 30),
                                         ((threadIdx.x & 4u) << 13) | ((threadIdx.x & 8u) << 28));
    }
    __syncthreads();

    const block_iq3_s * base[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        base[r] = (const block_iq3_s *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row + g;
    }

    float sum[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            sum[r][c] = 0.0f;
        }
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint16_t q[4][R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint8_t * p = base[r][sb - g].qs + 8 * t;
            q[0][r] = *(const uint16_t *)(p + 0);
            q[1][r] = *(const uint16_t *)(p + 2);
            q[2][r] = *(const uint16_t *)(p + 4);
            q[3][r] = *(const uint16_t *)(p + 6);
        }
        // The same 32 activations of a unit feed all R rows, so they are read ONCE per half unit and kept in
        // registers (2 x uint4 per column, 8 half2). Reading them inside the row loop instead repeated the whole
        // L1 stream R times, and at 4 bytes of y per half2 FMA that stream - not the weights - was the limit.
        // Half units keep the live set small enough to stay well under the 128-register occupancy cliff.
#pragma unroll
        for (int h = 0; h < 2; h++) {
            uint4 Y[NC][2];
            float yscale[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + sb * QK_K + 32 * t + 16 * h);
                Y[c][0] = __ldg(yp);
                Y[c][1] = __ldg(yp + 1);
                yscale[c] = __ldg(ys + (int64_t) c * (K / 32) + sb * 8 + t).x;
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const block_iq3_s * b = base[r] + (sb - g);
                const int       qh = b->qh[t];
                const uint8_t * sg = b->signs + 4 * t;
                const uint32_t lo = (uint32_t) q[0][r] | ((uint32_t) q[1][r] << 16);
                const uint32_t hi = (uint32_t) q[2][r] | ((uint32_t) q[3][r] << 16);

                uint32_t wv[8];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int      l     = 4*h + k;
                    const int      qbyte = (l < 4) ? ((lo >> (8*l)) & 0xFF) : ((hi >> (8*(l-4))) & 0xFF);
                    const int      idx   = qbyte | ((qh << (8 - l)) & 0x100);
                    const uint2    gp    = s_grid[idx];
                    const uint8_t  sb8   = sg[l >> 1];
                    const uint2    sp    = s_sign[(l & 1) ? (sb8 >> 4) : (sb8 & 0x0F)];
                    // low half2 takes bytes 0 and 1, high half2 bytes 2 and 3, in y order
                    wv[2*k]     = gp.x | sp.x;
                    wv[2*k + 1] = gp.y | sp.y;
                }
                const uint8_t sc = b->scales[t >> 1];
                const int     sn = (t & 1) ? (sc >> 4) : (sc & 0xF);
                // the packed half2 weights read (v << 6) * 2^-24, i.e. v * 2^-18, so dl undoes the 2^6 amplitude
                // and the per-column ys.x (2^(24-k), see gp100_prep_act) undoes the activation scaling
                const float   dl = __half2float(b->d) * (float) (1 + 2*sn) * (1.0f / 64.0f);
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const half2 * Yh = (const half2 *) Y[c];
                    half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        acc = __hfma2(*(const half2 *) &wv[i], Yh[i], acc);
                    }
                    sum[r][c] += dl * yscale[c] * __half2float(__hadd(__low2half(acc), __high2half(acc)));
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = sum[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) c * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// IQ3_XXS: 98-byte super-block (d, 64 bytes of 8-bit grid indices, 32 bytes of 4x7-bit ksigns + 4-bit scale).
// 98 is not a multiple of 4, so every field is 2-byte aligned at best. Unit t of a block owns qs[8t..8t+7]
// (8 grid indices) and the 4 aux bytes at qs[64+4t] (four 7-bit sign indices + the 4-bit scale).
// Each grid index expands to 4 weights, so 8 indices = the 32 weights that match y[sb*256 + 32t .. +31].
// The 256-entry grid table is staged in shared memory (1 KB); read from global it costs ~29% here.
// Standalone probe at 8192x8192 (98-byte blocks): naive 10.7, this layout R=1 145, R=2 206.6, R=4 169 GB/s,
// pure-read ceiling 533 GB/s. Ablations: dropping the signs 237, dropping the grid lookup 266, both 298.
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_iq3_xxs(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R; // with GLU: R rows of ffn_up, then the same R rows of ffn_gate
    GGML_UNUSED(s16);

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    __shared__ uint32_t s_grid[256];
    for (int i = threadIdx.x; i < 256; i += 128) {
        s_grid[i] = iq3xxs_grid[i];
    }
    __syncthreads();

    const block_iq3_xxs * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const block_iq3_xxs * W = (const block_iq3_xxs *) ((GLU && r >= R) ? vG : vW);
        const int row = min(row0 + r % R, nrows - 1);
        base[r] = W + (int64_t) row * stride_row + g;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nsb; sb += 4) {
        const half * yb = y + sb * QK_K + 32 * t;
        // y carries a per-32-block scale of 2^k (see gp100_prep_act): ys.x = 2^(24-k) undoes it after the
        // dot product, so every block sum needs a further 2^-24. Neither IQ3_S nor IQ3_XXS offsets its
        // weights by +128, so the block sum in ys.y goes unused.
        const float yscale = __ldg(ys + sb * 8 + t).x * (1.0f / 16777216.0f);
        // 32 weights = 64 bytes = 4 x uint4: pull the activations into registers once
        const uint4 yv0 = __ldg((const uint4 *)(yb +  0));
        const uint4 yv1 = __ldg((const uint4 *)(yb +  8));
        const uint4 yv2 = __ldg((const uint4 *)(yb + 16));
        const uint4 yv3 = __ldg((const uint4 *)(yb + 24));
        const uint32_t yw[16] = {yv0.x, yv0.y, yv0.z, yv0.w,
                                 yv1.x, yv1.y, yv1.z, yv1.w,
                                 yv2.x, yv2.y, yv2.z, yv2.w,
                                 yv3.x, yv3.y, yv3.z, yv3.w};
        uint16_t q[4][RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t * p = base[r][sb - g].qs + 8 * t;
            q[0][r] = *(const uint16_t *)(p + 0);
            q[1][r] = *(const uint16_t *)(p + 2);
            q[2][r] = *(const uint16_t *)(p + 4);
            q[3][r] = *(const uint16_t *)(p + 6);
        }
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const block_iq3_xxs * b = base[r] + (sb - g);
            const uint32_t lo = (uint32_t) q[0][r] | ((uint32_t) q[1][r] << 16);
            const uint32_t hi = (uint32_t) q[2][r] | ((uint32_t) q[3][r] << 16);
            const uint32_t aux = (uint32_t) *(const uint16_t *)(b->qs + 64 + 4*t)
                               | ((uint32_t) *(const uint16_t *)(b->qs + 66 + 4*t) << 16);

            float acc = 0.0f;
#pragma unroll
            for (int l = 0; l < 4; l++) {
                // 7-bit index: the eighth sign bit is fixed up so the eight signs multiply to +1
                const uint32_t v = (aux >> (7*l)) & 127;
                const uint32_t s = v ^ ((__popc(v) & 1u) << 7);
#pragma unroll
                for (int p = 0; p < 2; p++) {
                    const int      qb  = 2*l + p;
                    const int      idx = (qb < 4) ? ((lo >> (8*qb)) & 0xFF) : ((hi >> (8*(qb-4))) & 0xFF);
                    const uint32_t gv  = s_grid[idx];
                    const int      nib = (s >> (4*p)) & 0xF;
                    const uint32_t yp0 = yw[4*l + 2*p];
                    const uint32_t yp1 = yw[4*l + 2*p + 1];
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int w = (gv >> (8*k)) & 0xFF;
                        if ((nib >> k) & 1) w = -w;
                        const uint32_t ypair = (k < 2) ? yp0 : yp1;
                        const uint32_t yh    = (k & 1) ? (ypair >> 16) : (ypair & 0xFFFF);
                        acc = fmaf((float) w, __half2float(__ushort_as_half((unsigned short) yh)), acc);
                    }
                }
            }
            // scale: (0.5 + ls) / 2, applied once per 32 weights
            sum[r] += __half2float(b->d) * (0.5f + (float)(aux >> 28)) * 0.5f * yscale * acc;
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// IQ3_XXS with 2..8 columns: same packing as the IQ3_S kernel above, but this grid holds up to 0x3e (62)
// per byte rather than 0x0f, so the shift is 4 (62 << 4 = 992, still inside the 1023-wide subnormal
// mantissa) and dl divides by 2^4 instead of 2^6. The sign source differs too: one 7-bit index per four
// weights, with the eighth sign bit forced so the four signs multiply to +1.
template <int R, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_iq3_xxs_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    GGML_UNUSED(ys);
    const int K = nsb * QK_K;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + threadIdx.x / WARP_SIZE) * R;
    const int g = lane / 8;
    const int t = lane % 8;

    __shared__ uint32_t s_grid[256];
    for (int i = threadIdx.x; i < 256; i += 128) {
        s_grid[i] = iq3xxs_grid[i];
    }
    __syncthreads();

    const block_iq3_xxs * base[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        base[r] = (const block_iq3_xxs *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row + g;
    }

    float sum[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            sum[r][c] = 0.0f;
        }
    }

    for (int sb = g; sb < nsb; sb += 4) {
        uint16_t q[4][R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint8_t * p = base[r][sb - g].qs + 8 * t;
            q[0][r] = *(const uint16_t *)(p + 0);
            q[1][r] = *(const uint16_t *)(p + 2);
            q[2][r] = *(const uint16_t *)(p + 4);
            q[3][r] = *(const uint16_t *)(p + 6);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const block_iq3_xxs * b = base[r] + (sb - g);
            const uint32_t lo  = (uint32_t) q[0][r] | ((uint32_t) q[1][r] << 16);
            const uint32_t hi  = (uint32_t) q[2][r] | ((uint32_t) q[3][r] << 16);
            const uint32_t aux = (uint32_t) *(const uint16_t *)(b->qs + 64 + 4*t)
                               | ((uint32_t) *(const uint16_t *)(b->qs + 66 + 4*t) << 16);

            uint32_t wv[16];
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const uint32_t v = (aux >> (7*l)) & 127;
                const uint32_t s = v ^ ((__popc(v) & 1u) << 7);
#pragma unroll
                for (int p = 0; p < 2; p++) {
                    const int      qb  = 2*l + p;
                    const int      idx = (qb < 4) ? ((lo >> (8*qb)) & 0xFF) : ((hi >> (8*(qb-4))) & 0xFF);
                    const uint32_t gv  = s_grid[idx];
                    const uint32_t nib = (s >> (4*p)) & 0xF;
                    wv[2*qb]     = (((gv      ) & 0xFFu) <<  4) | (((gv >>  8) & 0xFFu) << 20) |
                                   ((nib & 1) << 15) | ((nib & 2) << 30);
                    wv[2*qb + 1] = (((gv >> 16) & 0xFFu) <<  4) | (((gv >> 24) & 0xFFu) << 20) |
                                   ((nib & 4) << 13) | ((nib & 8) << 28);
                }
            }
            // (0.5 + ls) / 2 per 32 weights; dl also undoes the 2^4 of the packed weights
            const float dl = __half2float(b->d) * (0.5f + (float)(aux >> 28)) * 0.5f * (1.0f / 16.0f);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + sb * QK_K + 32 * t);
                const uint4 yv[4] = {__ldg(yp), __ldg(yp + 1), __ldg(yp + 2), __ldg(yp + 3)};
                const float yscale = __ldg(ys + (int64_t) c * (K / 32) + sb * 8 + t).x;
                const half2 * Y = (const half2 *) yv;
                half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    acc = __hfma2(*(const half2 *) &wv[i], Y[i], acc);
                }
                sum[r][c] += dl * yscale * __half2float(__hadd(__low2half(acc), __high2half(acc)));
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = sum[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) c * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}
static __device__ __forceinline__ float gp100_s8_f(const uint32_t w, const int sel) {
    return __int_as_float(__byte_perm(w ^ 0x80808080u, 0x4B000000u, 0x7650 | sel)) - 8388736.f;
}

// Q6_K: 210-byte blocks are only 2-byte aligned, so 32-bit loads straight from global memory are both
// scattered and many (13 per thread per block: 329 GB/s). Instead, a warp's 4 consecutive blocks of a row
// (840 contiguous bytes) are staged with coalesced 16-byte loads into a per-warp shared buffer, and each
// thread reads its 8-byte chunks from there with a funnel shift (8704x5120: 380 GB/s vs 246 generic).
// Thread t of a group owns half n = t/4 and positions l = 8*(t%4)..+7 of all four 32-runs of that half,
// i.e. 8 weights of each of 4 sixteen-wide sub-blocks. q (0..63) is used as a subnormal half; the -32
// offset is removed in fp32 with the per-16 sums, once per sub-block by the even thread of each pair.
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_q6_K(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R; // with GLU: R rows of ffn_up, then the same R rows of ffn_gate
    constexpr int STAGE_U4 = 56; // >= (15 + 4*210 + 15) / 16
    __shared__ uint4 stage[4][RT][STAGE_U4];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    const int g = lane / 8;
    const int t = lane % 8;
    const int n = t / 4;
    const int c = t % 4;
    const int i0 = c / 2;
    const float offm = (c & 1) ? 0.0f : -32.0f;

    const uint4 * gp[RT];
    int mis[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const uintptr_t a = (uintptr_t) ((const block_q6_K *) ((GLU && r >= R) ? vG : vW) + (int64_t) min(row0 + r % R, nrows - 1) * stride_row);
        gp[r]  = (const uint4 *) (a & ~(uintptr_t) 15);
        mis[r] = a & 15;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    const int nit = (nsb + 3) / 4;
    for (int it = 0; it < nit; it++) {
        const int  sb   = 4 * it + g;
        const bool act  = sb < nsb;
        const int  nblk = min(nsb - 4 * it, 4);

        int m[RT];
        uint4 v0[RT], v1[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const int off = mis[r] + 840 * it;
            const uint4 * src = gp[r] + (off >> 4);
            m[r] = off & 15;
            const int nu4 = (m[r] + 210 * nblk + 15) >> 4; // only the 16-byte chunks that hold these blocks
            v0[r] = lane      < nu4 ? __ldg(src + lane)      : make_uint4(0, 0, 0, 0);
            v1[r] = lane + 32 < nu4 ? __ldg(src + lane + 32) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < RT; r++) {
            stage[warp][r][lane] = v0[r];
            if (lane + 32 < STAGE_U4) {
                stage[warp][r][lane + 32] = v1[r];
            }
        }
        __syncwarp();

        const uint4 zero4 = make_uint4(0, 0, 0, 0);
        const uint4 * yp = (const uint4 *) (y + sb * 256 + 128 * n + 8 * c);
        const uint4 yv[4] = {act ? __ldg(yp) : zero4, act ? __ldg(yp + 4) : zero4, act ? __ldg(yp + 8) : zero4, act ? __ldg(yp + 12) : zero4};
        const float4 rs01 = act ? __ldg((const float4 *) (ys + sb * 8 + 4 * n))     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 rs23 = act ? __ldg((const float4 *) (ys + sb * 8 + 4 * n + 2)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float  rs[4] = {rs01.x, rs01.z, rs23.x, rs23.z};
        const float * sp = s16 + sb * 16 + 8 * n + i0;
        const float  so[4] = {act ? __ldg(sp + 0) * offm : 0.0f, act ? __ldg(sp + 2) * offm : 0.0f,
                              act ? __ldg(sp + 4) * offm : 0.0f, act ? __ldg(sp + 6) * offm : 0.0f};

#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + m[r] + 210 * g; // this group's block
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;
            auto ld8 = [&](const int byteoff) {
                const uint32_t * q = w + byteoff / 4;
                return make_uint2(__funnelshift_r(q[0], q[1], shf), __funnelshift_r(q[1], q[2], shf));
            };
            const uint2    L0 = ld8(64 * n + 8 * c);
            const uint2    L1 = ld8(64 * n + 32 + 8 * c);
            const uint2    QH = ld8(128 + 32 * n + 8 * c);
            const uint2    SC = ld8(192 + 8 * n);
            const uint32_t DW = __funnelshift_r(w[52], w[53], shf); // d at byte 208

            half2 acc[4];
#pragma unroll
            for (int mm = 0; mm < 4; mm++) {
                acc[mm] = __float2half2_rn(0.0f);
            }
            const uint32_t l0w[2] = {L0.x, L0.y};
            const uint32_t l1w[2] = {L1.x, L1.y};
            const uint32_t hw[2]  = {QH.x, QH.y};
#pragma unroll
            for (int k = 0; k < 2; k++) { // 4 bytes = positions 8c+4k .. +3
                const uint32_t a = l0w[k], b = l1w[k], h = hw[k];
                const uint32_t v[4] = {
                    ( a       & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4),
                    ( b       & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2),
                    ((a >> 4) & 0x0F0F0F0Fu) | ( h & 0x30303030u),
                    ((b >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u)};
#pragma unroll
                for (int mm = 0; mm < 4; mm++) {
                    const half2 * Y = (const half2 *) &yv[mm];
                    const uint32_t p0 = __byte_perm(v[mm], 0, 0x4140);
                    const uint32_t p1 = __byte_perm(v[mm], 0, 0x4342);
                    acc[mm] = __hfma2(*(const half2 *) &p0, Y[2*k + 0], acc[mm]);
                    acc[mm] = __hfma2(*(const half2 *) &p1, Y[2*k + 1], acc[mm]);
                }
            }
            // scales 8n + c/2 + {0,2,4,6}: bytes c/2 and c/2+2 of each 4-byte word
            const float sc[4] = {gp100_s8_f(SC.x, i0), gp100_s8_f(SC.x, i0 + 2), gp100_s8_f(SC.y, i0), gp100_s8_f(SC.y, i0 + 2)};
            float s = 0.0f;
#pragma unroll
            for (int mm = 0; mm < 4; mm++) {
                s = fmaf(sc[mm], fmaf(__half2float(__hadd(__low2half(acc[mm]), __high2half(acc[mm]))), rs[mm], so[mm]), s);
            }
            if (act) {
                sum[r] = fmaf(__half2float(__ushort_as_half((unsigned short) (DW & 0xFFFF))), s, sum[r]);
            }
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q6_K with 2..8 columns: same staging and unpack as gp100_mmvq_q6_K, one accumulator set per column.
template <int R, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_q6_K_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    constexpr int RT = R;
    const int K = nsb * QK_K;
    constexpr int STAGE_U4 = 56; // >= (15 + 4*210 + 15) / 16
    __shared__ uint4 stage[4][RT][STAGE_U4];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    const int g = lane / 8;
    const int t = lane % 8;
    const int n = t / 4;
    const int c4 = t % 4;
    const int i0 = c4 / 2;
    const float offm = (c4 & 1) ? 0.0f : -32.0f;

    const uint4 * gp[RT];
    int mis[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const uintptr_t a = (uintptr_t) ((const block_q6_K *) vW + (int64_t) min(row0 + r % R, nrows - 1) * stride_row);
        gp[r]  = (const uint4 *) (a & ~(uintptr_t) 15);
        mis[r] = a & 15;
    }

    float sum[RT][NC];
#pragma unroll
    for (int r = 0; r < RT; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            sum[r][c] = 0.0f;
        }
    }

    const int nit = (nsb + 3) / 4;
    for (int it = 0; it < nit; it++) {
        const int  sb   = 4 * it + g;
        const bool act  = sb < nsb;
        // activation reads use a clamped super-block so they are unconditional (identical for every row, so the
        // compiler loads them once per column rather than once per row); act gates only the accumulation
        const int  sbc  = act ? sb : nsb - 1;
        const int  nblk = min(nsb - 4 * it, 4);

        int m[RT];
        uint4 v0[RT], v1[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const int off = mis[r] + 840 * it;
            const uint4 * src = gp[r] + (off >> 4);
            m[r] = off & 15;
            const int nu4 = (m[r] + 210 * nblk + 15) >> 4; // only the 16-byte chunks that hold these blocks
            v0[r] = lane      < nu4 ? __ldg(src + lane)      : make_uint4(0, 0, 0, 0);
            v1[r] = lane + 32 < nu4 ? __ldg(src + lane + 32) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < RT; r++) {
            stage[warp][r][lane] = v0[r];
            if (lane + 32 < STAGE_U4) {
                stage[warp][r][lane + 32] = v1[r];
            }
        }
        __syncwarp();

        // per-column scales and -32 offsets do not depend on the row: load them once per iteration
        float rsc[NC][4], soc[NC][4];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float2 * ysc = ys + (int64_t) c * (K / 32);
            const float4 rs01 = __ldg((const float4 *) (ysc + sbc * 8 + 4 * n));
            const float4 rs23 = __ldg((const float4 *) (ysc + sbc * 8 + 4 * n + 2));
            rsc[c][0] = rs01.x; rsc[c][1] = rs01.z; rsc[c][2] = rs23.x; rsc[c][3] = rs23.z;
            const float * sp = s16 + (int64_t) c * (K / 16) + sbc * 16 + 8 * n + i0;
#pragma unroll
            for (int mm = 0; mm < 4; mm++) {
                soc[c][mm] = __ldg(sp + 2 * mm) * offm;
            }
        }
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + m[r] + 210 * g; // this group's block
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;
            auto ld8 = [&](const int byteoff) {
                const uint32_t * q = w + byteoff / 4;
                return make_uint2(__funnelshift_r(q[0], q[1], shf), __funnelshift_r(q[1], q[2], shf));
            };
            const uint2    L0 = ld8(64 * n + 8 * c4);
            const uint2    L1 = ld8(64 * n + 32 + 8 * c4);
            const uint2    QH = ld8(128 + 32 * n + 8 * c4);
            const uint2    SC = ld8(192 + 8 * n);
            const uint32_t DW = __funnelshift_r(w[52], w[53], shf);

            uint32_t wv[4][4]; // [run mm][2k + half]: this row's 32 weights as subnormal half2
            const uint32_t l0w[2] = {L0.x, L0.y};
            const uint32_t l1w[2] = {L1.x, L1.y};
            const uint32_t hw[2]  = {QH.x, QH.y};
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const uint32_t a = l0w[k], b = l1w[k], h = hw[k];
                const uint32_t v[4] = {
                    ( a       & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4),
                    ( b       & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2),
                    ((a >> 4) & 0x0F0F0F0Fu) | ( h & 0x30303030u),
                    ((b >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u)};
#pragma unroll
                for (int mm = 0; mm < 4; mm++) {
                    wv[mm][2*k]     = __byte_perm(v[mm], 0, 0x4140);
                    wv[mm][2*k + 1] = __byte_perm(v[mm], 0, 0x4342);
                }
            }
            const float sc[4] = {gp100_s8_f(SC.x, i0), gp100_s8_f(SC.x, i0 + 2), gp100_s8_f(SC.y, i0), gp100_s8_f(SC.y, i0 + 2)};
            const float dd = __half2float(__ushort_as_half((unsigned short) (DW & 0xFFFF)));
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + sbc * 256 + 128 * n + 8 * c4);
                const uint4 yv[4] = {__ldg(yp), __ldg(yp + 4), __ldg(yp + 8), __ldg(yp + 12)};
                float sacc = 0.0f;
#pragma unroll
                for (int mm = 0; mm < 4; mm++) {
                    const half2 * Y = (const half2 *) &yv[mm];
                    half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        acc = __hfma2(*(const half2 *) &wv[mm][i], Y[i], acc);
                    }
                    sacc = fmaf(sc[mm], fmaf(__half2float(__hadd(__low2half(acc), __high2half(acc))), rsc[c][mm], soc[c][mm]), sacc);
                }
                if (act) {
                    sum[r][c] = fmaf(dd, sacc, sum[r][c]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = sum[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) c * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q3_K: 110-byte blocks (hmask[32], qs[64], scales[12], d), 16 sub-blocks of 16 with signed 6-bit scales.
// Weight 128n + 32j + l = ((qs[32n+l] >> 2j) & 3) + 4 * (hmask[l] >> (4n+j) & 1) - 4. Staged like Q6_K
// (a warp's 4 consecutive blocks = 440 contiguous bytes, one 16-byte load per lane); thread t of a group owns
// half n = t/4 and positions l = 8*(t%4)..+7 of the four 32-runs of that half. q + 4 (0..7) is used as a
// subnormal half; the -4 is removed in fp32 with the per-16 sums, once per sub-block (even thread of a pair).
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_q3_K(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R;
    constexpr int BS = sizeof(block_q3_K); // 110
    constexpr int STAGE_U4 = (15 + 4*BS + 15) / 16 + 1; // + 1: funnel-shifted reads may touch one word past the end
    __shared__ uint4 stage[4][RT][STAGE_U4];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    const int g = lane / 8;
    const int t = lane % 8;
    const int n = t / 4;
    const int c = t % 4;
    const int i0 = c / 2;
    const float offm = (c & 1) ? 0.0f : -4.0f;
    // scale words for this half: n=0 -> sub-blocks 0..3 and 4..7, n=1 -> 8..11 and 12..15
    const int sA  = 4 * n;     // low nibble source: A (n=0) or A >> 4 (n=1); same for B
    const int sC0 = 4 * n;     // high 2 bits of sub-blocks 0..3 / 8..11 from C >> (0 / 4)
    const int sC1 = 2 + 4 * n; // high 2 bits of sub-blocks 4..7 / 12..15 from C >> (2 / 6)

    const uint4 * gp[RT];
    int mis[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const uintptr_t a = (uintptr_t) ((const block_q3_K *) ((GLU && r >= R) ? vG : vW) + (int64_t) min(row0 + r % R, nrows - 1) * stride_row);
        gp[r]  = (const uint4 *) (a & ~(uintptr_t) 15);
        mis[r] = a & 15;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    const int nit = (nsb + 3) / 4;
    for (int it = 0; it < nit; it++) {
        const int  sb   = 4 * it + g;
        const bool act  = sb < nsb;
        const int  nblk = min(nsb - 4 * it, 4);

        int m[RT];
        uint4 v0[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const int off = mis[r] + 4 * BS * it;
            m[r] = off & 15;
            const int nu4 = (m[r] + BS * nblk + 15) >> 4;
            v0[r] = lane < nu4 ? __ldg(gp[r] + (off >> 4) + lane) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < RT; r++) {
            if (lane < STAGE_U4) {
                stage[warp][r][lane] = v0[r];
            }
        }
        __syncwarp();

        const uint4 zero4 = make_uint4(0, 0, 0, 0);
        const uint4 * yp = (const uint4 *) (y + sb * 256 + 128 * n + 8 * c);
        const uint4 yv[4] = {act ? __ldg(yp) : zero4, act ? __ldg(yp + 4) : zero4, act ? __ldg(yp + 8) : zero4, act ? __ldg(yp + 12) : zero4};
        const float4 rs01 = act ? __ldg((const float4 *) (ys + sb * 8 + 4 * n))     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 rs23 = act ? __ldg((const float4 *) (ys + sb * 8 + 4 * n + 2)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float  rs[4] = {rs01.x, rs01.z, rs23.x, rs23.z};
        const float * sp = s16 + sb * 16 + 8 * n + i0;
        const float  so[4] = {act ? __ldg(sp + 0) * offm : 0.0f, act ? __ldg(sp + 2) * offm : 0.0f,
                              act ? __ldg(sp + 4) * offm : 0.0f, act ? __ldg(sp + 6) * offm : 0.0f};

#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + m[r] + BS * g; // this group's block
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;
            auto ld4 = [&](const int byteoff) { // 4 bytes at a 2-aligned offset
                const uint32_t * q = w + byteoff / 4;
                return __funnelshift_r(q[0], q[1], shf);
            };
            const uint32_t QS[2] = {ld4(32 + 32 * n + 8 * c), ld4(32 + 32 * n + 8 * c + 4)};
            const uint32_t HM[2] = {ld4(8 * c), ld4(8 * c + 4)};
            const uint32_t A = ld4(96), B = ld4(100), C = ld4(104);
            const uint32_t DW = ld4(108);
            const uint32_t W0 = ((A >> sA) & 0x0F0F0F0Fu) | (((C >> sC0) & 0x03030303u) << 4); // sub-blocks 8n + 0..3
            const uint32_t W1 = ((B >> sA) & 0x0F0F0F0Fu) | (((C >> sC1) & 0x03030303u) << 4); // sub-blocks 8n + 4..7

            half2 acc[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] = __float2half2_rn(0.0f);
            }
#pragma unroll
            for (int k = 0; k < 2; k++) { // 4 bytes = positions 8c+4k .. +3
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t v  = ((QS[k] >> (2 * j)) & 0x03030303u) | (((HM[k] >> (4 * n + j)) & 0x01010101u) << 2);
                    const half2 * Y   = (const half2 *) &yv[j];
                    const uint32_t p0 = __byte_perm(v, 0, 0x4140);
                    const uint32_t p1 = __byte_perm(v, 0, 0x4342);
                    acc[j] = __hfma2(*(const half2 *) &p0, Y[2*k + 0], acc[j]);
                    acc[j] = __hfma2(*(const half2 *) &p1, Y[2*k + 1], acc[j]);
                }
            }
            // sub-block 8n + 2j + c/2: j=0,1 -> bytes i0, i0+2 of W0; j=2,3 -> bytes i0, i0+2 of W1
            const float sc[4] = {gp100_byte_f(W0, i0) - 32.0f, gp100_byte_f(W0, i0 + 2) - 32.0f,
                                 gp100_byte_f(W1, i0) - 32.0f, gp100_byte_f(W1, i0 + 2) - 32.0f};
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s = fmaf(sc[j], fmaf(__half2float(__hadd(__low2half(acc[j]), __high2half(acc[j]))), rs[j], so[j]), s);
            }
            if (act) {
                sum[r] = fmaf(__half2float(__ushort_as_half((unsigned short) (DW & 0xFFFF))), s, sum[r]);
            }
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q2_K: an 84-byte super-block of 16 packed scale/min nibbles, 64 bytes of 2-bit quants, then d and dmin.
// dequantize_row_q2_K walks it as two 128-value halves, each read at four shifts (0, 2, 4, 6) over the same
// 32 quant bytes, so thread t of a group of eight owns qs[32*(t/4) .. +31] at shift 2*(t%4): exactly two
// 16-value runs, which differ only in their scale nibble. Both 2-bit digits and the 4-bit scales are used
// as-is; the digits go into subnormal halves like the 4-bit types (3 << 6 = 192 is exact), and that 2^6 is
// folded into `rescale` below. The 16-value runs line up with s16, so the min term needs no interpolation.
template <int R, bool GLU>
static __global__ void __launch_bounds__(128) gp100_mmvq_q2_K(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys,
        const float * __restrict__ s16, float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    constexpr int RT = GLU ? 2*R : R;
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * R;
    const int g    = lane / 8;
    const int t    = lane % 8;
    const int shf  = 2 * (t % 4);         // this thread's 2-bit shift
    const int qb   = 32 * (t / 4);        // first or second half of the 64 quant bytes
    const int is   = 8 * (t / 4) + 2 * (t % 4); // index of its first scale nibble (= first 16-value run)

    const block_q2_K * base[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        const block_q2_K * W = (const block_q2_K *) ((GLU && r >= R) ? vG : vW);
        const int row = min(row0 + r % R, nrows - 1);
        base[r] = W + (int64_t) row * stride_row + g;
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int sb = g; sb < nsb; sb += 4) {
        // the 32 activations these two runs consume, read once for every row in flight
        const half * yb = y + sb * QK_K + 16 * is;
        const uint4 yv0 = __ldg((const uint4 *) (yb +  0));
        const uint4 yv1 = __ldg((const uint4 *) (yb +  8));
        const uint4 yv2 = __ldg((const uint4 *) (yb + 16));
        const uint4 yv3 = __ldg((const uint4 *) (yb + 24));
        const uint32_t yw[16] = {yv0.x, yv0.y, yv0.z, yv0.w, yv1.x, yv1.y, yv1.z, yv1.w,
                                 yv2.x, yv2.y, yv2.z, yv2.w, yv3.x, yv3.y, yv3.z, yv3.w};
        // ys.x = 2^(24-k) for the 32-block holding these runs; a weight digit is 2^-18 at half scale
        // (v << 6), so the rescale carries an extra 2^-6.
        const float rescale = __ldg(ys + sb * 8 + (is >> 1)).x * (1.0f / 64.0f);
        const float sx0 = __ldg(s16 + sb * 16 + is);
        const float sx1 = __ldg(s16 + sb * 16 + is + 1);
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const block_q2_K * b = base[r] + (sb - g);
            const uint32_t * qp = (const uint32_t *) (b->qs + qb);
            const uint32_t sc2 = *(const uint16_t *) (b->scales + is); // scales[is], scales[is + 1]
            half2 a0 = __float2half2_rn(0.0f);
            half2 a1 = a0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t w0 = __ldg(qp + k);
                const uint32_t w1 = __ldg(qp + 4 + k);
                const uint32_t v0 = (w0 >> shf) & 0x03030303u;
                const uint32_t v1 = (w1 >> shf) & 0x03030303u;
                const uint32_t p0 = __byte_perm(v0, 0, 0x4140) << 6;
                const uint32_t p1 = __byte_perm(v0, 0, 0x4342) << 6;
                const uint32_t p2 = __byte_perm(v1, 0, 0x4140) << 6;
                const uint32_t p3 = __byte_perm(v1, 0, 0x4342) << 6;
                a0 = __hfma2(*(const half2 *) &p0, *(const half2 *) &yw[2*k + 0], a0);
                a0 = __hfma2(*(const half2 *) &p1, *(const half2 *) &yw[2*k + 1], a0);
                a1 = __hfma2(*(const half2 *) &p2, *(const half2 *) &yw[8 + 2*k + 0], a1);
                a1 = __hfma2(*(const half2 *) &p3, *(const half2 *) &yw[8 + 2*k + 1], a1);
            }
            const float dA = __half2float(__hadd(__low2half(a0), __high2half(a0)));
            const float dB = __half2float(__hadd(__low2half(a1), __high2half(a1)));
            const float2 dm = __half22float2(b->dm); // d, dmin (an anonymous union of the two halves)
            sum[r] += dm.x * rescale * fmaf((float) ( sc2        & 0x0F), dA, (float) ((sc2 >> 8) & 0x0F) * dB)
                    - dm.y *           fmaf((float) ((sc2 >> 4) & 0x0F), sx0, (float) ((sc2 >> 12) & 0x0F) * sx1);
        }
    }

    gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Blocks of 32 weights with one fp16 scale: Q8_0 (34 bytes) and IQ4_NL (18 bytes). These take K % 32 rather
// than K % QK_K, so the block count is passed as 32-blocks; every run of 32 blocks is a whole number of uint4
// (1088 / 576 bytes for Q8_0 at 34 bytes a block, 576 / 288 for IQ4_NL at 18), so a warp stages 32 blocks of a
// row with aligned 16-byte loads and each lane takes one block from shared memory (2-byte aligned there,
// read with funnel shifts). A row itself must still be 16-byte aligned. Values are used +128 as unsigned subnormal halves (Q8_0: q ^ 0x80; IQ4_NL: the
// kvalues_iq4nl table stored +128, looked up like IQ4_XS); the 128 * sum(x) this adds is removed in fp32.
template <ggml_type type> struct gp100_b32;
template <> struct gp100_b32<GGML_TYPE_Q8_0>   { typedef block_q8_0   block; };
template <> struct gp100_b32<GGML_TYPE_IQ4_NL> { typedef block_iq4_nl block; };

// KS: how many of the block's four warps cooperate on one row group, each taking every KS-th chunk of K.
// KS = 1 is one warp per R rows over the whole K; KS = 4 shortens that chain fourfold for short matrices.
template <ggml_type type, int R, bool GLU, int KS>
static __global__ void __launch_bounds__(128) gp100_mmvq_b32(
        const void * __restrict__ vW, const void * __restrict__ vG, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row,
        const int32_t * __restrict__ ids, const int64_t stride_xpert, const int64_t stride_y, const int64_t stride_dst, const int n32,
        const int n_slots, const int ybc, const int stride_ids) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    {
        const gp100_pairs P = gp100_pair_rebase({vW, y, ys, s16, dst}, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
        vW = P.W; y = P.y; ys = P.ys; s16 = P.s16; dst = P.dst;
    }
    constexpr int RT = GLU ? 2*R : R;
    typedef typename gp100_b32<type>::block block;
    constexpr int BS     = sizeof(block);
    constexpr int SEG_U4 = 32 * BS / sizeof(uint4); // Q8_0 68, IQ4_NL 36
    static_assert(32 * BS % sizeof(uint4) == 0, "32 blocks must be a whole number of uint4");
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u; // kvalues_iq4nl + 128
    __shared__ uint4 stage[4][RT][SEG_U4 + 1]; // +1: the funnel-shifted reads of lane 31 touch one word past its block
    GGML_UNUSED(s16);

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * (4 / KS) + warp / KS) * R;
    const int nb   = nsb; // Q8_0/IQ4_NL blocks per row: the b32 kernels take 32-blocks, so K need not be a multiple of QK_K

    const uint4 * rowp[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        rowp[r] = (const uint4 *) ((const block *) ((GLU && r >= R) ? vG : vW) + (int64_t) min(row0 + r % R, nrows - 1) * stride_row);
    }

    float sum[RT];
#pragma unroll
    for (int r = 0; r < RT; r++) {
        sum[r] = 0.0f;
    }

    for (int c0 = warp % KS; c0 * 32 < nb; c0 += KS) {
        const int b0   = c0 * 32;
        const int nblk = min(nb - b0, 32);
        uint4 v0[RT], v1[RT], v2[RT];
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint4 * src = rowp[r] + b0 * BS / 16;
            const int nu4 = (nblk * BS + 15) / 16;
            v0[r] = lane      < nu4 ? __ldg(src + lane)      : make_uint4(0, 0, 0, 0);
            v1[r] = lane + 32 < nu4 ? __ldg(src + lane + 32) : make_uint4(0, 0, 0, 0);
            v2[r] = SEG_U4 > 64 && lane + 64 < nu4 ? __ldg(src + lane + 64) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < RT; r++) {
            stage[warp][r][lane] = v0[r];
            if (lane + 32 < SEG_U4) {
                stage[warp][r][lane + 32] = v1[r];
            }
            if (lane + 64 < SEG_U4) {
                stage[warp][r][lane + 64] = v2[r];
            }
        }
        __syncwarp();

        const int  b   = b0 + lane;
        const bool act = lane < nblk;
        const uint4 zero4 = make_uint4(0, 0, 0, 0);
        const uint4 * yp = (const uint4 *) (y + 32 * b);
        const uint4 yv[4] = {act ? __ldg(yp) : zero4, act ? __ldg(yp + 1) : zero4, act ? __ldg(yp + 2) : zero4, act ? __ldg(yp + 3) : zero4};
        const float2 sc = act ? __ldg(ys + b) : make_float2(0.0f, 0.0f); // {rescale, sum}
        const half2 * Y = (const half2 *) yv;

#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + BS * lane; // this lane's block
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;          // 0 or 16
            const half       d   = __ushort_as_half((unsigned short) (__funnelshift_r(w[0], w[1], shf) & 0xFFFF));
            // qs start 2 bytes after d: word offset (shf+16)/32, shift (shf+16)%32
            const int      qo  = (shf + 16) / 32;
            const uint32_t qsh = (shf + 16) % 32;
            half2 acc = __float2half2_rn(0.0f);
            if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const uint32_t q  = __funnelshift_r(w[qo + k], w[qo + k + 1], qsh) ^ 0x80808080u; // q + 128, per byte
                    const uint32_t p0 = __byte_perm(q, 0, 0x4140);
                    const uint32_t p1 = __byte_perm(q, 0, 0x4342);
                    acc = __hfma2(*(const half2 *) &p0, Y[2*k + 0], acc);
                    acc = __hfma2(*(const half2 *) &p1, Y[2*k + 1], acc);
                }
            } else { // IQ4_NL: qs[j] low nibble = weight j, high = weight j+16 (same as an IQ4_XS sub-block)
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint32_t q  = __funnelshift_r(w[qo + k], w[qo + k + 1], qsh);
                    const uint32_t qx = q ^ 0x88888888u;
                    const uint32_t S  = ((q >> 1) & 0x44444444u) | 0x32103210u;
                    const uint32_t v0 = gp100_prmt(gp100_prmt(T0, T1, q),       gp100_prmt(T2, T3, qx),       S);
                    const uint32_t v1 = gp100_prmt(gp100_prmt(T0, T1, q >> 16), gp100_prmt(T2, T3, qx >> 16), S >> 16);
                    const uint32_t pa = __byte_perm(v0, 0, 0x4240); // w(4k),    w(4k+1)
                    const uint32_t pb = __byte_perm(v1, 0, 0x4240); // w(4k+2),  w(4k+3)
                    const uint32_t pc = __byte_perm(v0, 0, 0x4341); // w(16+4k), w(17+4k)
                    const uint32_t pd = __byte_perm(v1, 0, 0x4341); // w(18+4k), w(19+4k)
                    acc = __hfma2(*(const half2 *) &pa, Y[2*k + 0],     acc);
                    acc = __hfma2(*(const half2 *) &pb, Y[2*k + 1],     acc);
                    acc = __hfma2(*(const half2 *) &pc, Y[8 + 2*k + 0], acc);
                    acc = __hfma2(*(const half2 *) &pd, Y[8 + 2*k + 1], acc);
                }
            }
            const float s = __half2float(__hadd(__low2half(acc), __high2half(acc)));
            if (act) {
                sum[r] = fmaf(__half2float(d), fmaf(s, sc.x, -128.0f * sc.y), sum[r]);
            }
        }
    }

    if constexpr (KS == 1) {
        gp100_store<R, GLU>(sum, dst, row0, nrows, lane);
    } else {
        gp100_store_ksplit<R, GLU>(sum, dst, row0, nrows, lane, warp);
    }
#else
    GGML_UNUSED_VARS(vW, vG, y, ys, s16, dst, nrows, nsb, stride_row, ids, stride_xpert, stride_y, stride_dst, n32, n_slots, ybc, stride_ids);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q8_0 (and IQ4_NL, not dispatched: slower than the generic kernel) with 2..8 columns: gp100_mmvq_b32's staging, each lane's block unpacked ONCE into 16 half2 registers,
// then every column FMA'd against them. Activation reads use a clamped block index so they are unconditional (and
// identical for every row, so they are loaded once per column; see gp100_mmvq_q6_K_nc); `act` gates only the sum.
template <ggml_type type, int R, int KS, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_b32_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    typedef typename gp100_b32<type>::block block;
    constexpr int BS     = sizeof(block);
    constexpr int SEG_U4 = 32 * BS / sizeof(uint4);
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u; // kvalues_iq4nl + 128
    __shared__ uint4 stage[4][R][SEG_U4 + 1];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * (4 / KS) + warp / KS) * R;
    const int nb   = nsb; // Q8_0/IQ4_NL blocks per row: the b32 kernels take 32-blocks, so K need not be a multiple of QK_K
    const int K    = nb * QK8_0;

    const uint4 * rowp[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        rowp[r] = (const uint4 *) ((const block *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row);
    }

    float sum[NC * R]; // [c][r]: the store helpers see NC*R rows, and dst row c*R + r is mapped below
#pragma unroll
    for (int i = 0; i < NC * R; i++) {
        sum[i] = 0.0f;
    }

    for (int c0 = warp % KS; c0 * 32 < nb; c0 += KS) {
        const int b0   = c0 * 32;
        const int nblk = min(nb - b0, 32);
        uint4 v0[R], v1[R], v2[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint4 * src = rowp[r] + b0 * BS / 16;
            const int nu4 = (nblk * BS + 15) / 16;
            v0[r] = lane      < nu4 ? __ldg(src + lane)      : make_uint4(0, 0, 0, 0);
            v1[r] = lane + 32 < nu4 ? __ldg(src + lane + 32) : make_uint4(0, 0, 0, 0);
            v2[r] = SEG_U4 > 64 && lane + 64 < nu4 ? __ldg(src + lane + 64) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; r++) {
            stage[warp][r][lane] = v0[r];
            if (lane + 32 < SEG_U4) {
                stage[warp][r][lane + 32] = v1[r];
            }
            if (lane + 64 < SEG_U4) {
                stage[warp][r][lane + 64] = v2[r];
            }
        }
        __syncwarp();

        const bool act = lane < nblk;
        const int  bc  = act ? b0 + lane : nb - 1;

#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + BS * lane;
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;
            const half       d   = __ushort_as_half((unsigned short) (__funnelshift_r(w[0], w[1], shf) & 0xFFFF));
            const int      qo  = (shf + 16) / 32;
            const uint32_t qsh = (shf + 16) % 32;
            uint32_t wv[16]; // this lane's 32 weights (+128) as subnormal half2, in y order
            if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const uint32_t q = __funnelshift_r(w[qo + k], w[qo + k + 1], qsh) ^ 0x80808080u;
                    wv[2*k + 0] = __byte_perm(q, 0, 0x4140);
                    wv[2*k + 1] = __byte_perm(q, 0, 0x4342);
                }
            } else {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint32_t q  = __funnelshift_r(w[qo + k], w[qo + k + 1], qsh);
                    const uint32_t qx = q ^ 0x88888888u;
                    const uint32_t S  = ((q >> 1) & 0x44444444u) | 0x32103210u;
                    const uint32_t v0 = gp100_prmt(gp100_prmt(T0, T1, q),       gp100_prmt(T2, T3, qx),       S);
                    const uint32_t v1 = gp100_prmt(gp100_prmt(T0, T1, q >> 16), gp100_prmt(T2, T3, qx >> 16), S >> 16);
                    wv[2*k + 0]     = __byte_perm(v0, 0, 0x4240);
                    wv[2*k + 1]     = __byte_perm(v1, 0, 0x4240);
                    wv[8 + 2*k + 0] = __byte_perm(v0, 0, 0x4341);
                    wv[8 + 2*k + 1] = __byte_perm(v1, 0, 0x4341);
                }
            }
            const float df = __half2float(d);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) c * K + 32 * bc);
                const uint4 yv[4] = {__ldg(yp), __ldg(yp + 1), __ldg(yp + 2), __ldg(yp + 3)};
                const float2 sc = __ldg(ys + (int64_t) c * nb + bc);
                const half2 * Y = (const half2 *) yv;
                half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    acc = __hfma2(*(const half2 *) &wv[i], Y[i], acc);
                }
                const float sv = __half2float(__hadd(__low2half(acc), __high2half(acc)));
                if (act) {
                    sum[c * R + r] = fmaf(df, fmaf(sv, sc.x, -128.0f * sc.y), sum[c * R + r]);
                }
            }
        }
    }

    // reduce like gp100_store / gp100_store_ksplit, then write column c of row r to dst[c * stride_dst + row0 + r]
#pragma unroll
    for (int i = 0; i < NC * R; i++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum[i] += __shfl_xor_sync(0xFFFFFFFF, sum[i], o);
        }
    }
    if constexpr (KS > 1) {
        __shared__ float red[4][NC * R];
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < NC * R; i++) {
                red[warp][i] = sum[i];
            }
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
#pragma unroll
        for (int i = 0; i < NC * R; i++) {
            sum[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (row0 + r < nrows) {
                    dst[(int64_t) c * stride_dst + row0 + r] = sum[c * R + r];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Q3_K with 2..8 columns: gp100_mmvq_q3_K's staging and unpack, each row's 32 weights per thread unpacked once,
// then every column FMA'd against them. Activation reads use a clamped super-block (unconditional, shared across
// rows); `act` gates only the accumulation.
template <int R, int NC>
static __global__ void __launch_bounds__(128) gp100_mmvq_q3_K_nc(
        const void * __restrict__ vW, const half * __restrict__ y, const float2 * __restrict__ ys, const float * __restrict__ s16,
        float * __restrict__ dst, const int nrows, const int nsb, const int stride_row, const int stride_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    constexpr int BS = sizeof(block_q3_K); // 110
    constexpr int STAGE_U4 = (15 + 4*BS + 15) / 16 + 1;
    __shared__ uint4 stage[4][R][STAGE_U4];
    const int K = nsb * QK_K;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    const int g = lane / 8;
    const int t = lane % 8;
    const int n = t / 4;
    const int c = t % 4;
    const int i0 = c / 2;
    const float offm = (c & 1) ? 0.0f : -4.0f;
    const int sA  = 4 * n;
    const int sC0 = 4 * n;
    const int sC1 = 2 + 4 * n;

    const uint4 * gp[R];
    int mis[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const uintptr_t a = (uintptr_t) ((const block_q3_K *) vW + (int64_t) min(row0 + r, nrows - 1) * stride_row);
        gp[r]  = (const uint4 *) (a & ~(uintptr_t) 15);
        mis[r] = a & 15;
    }

    float sum[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int cc = 0; cc < NC; cc++) {
            sum[r][cc] = 0.0f;
        }
    }

    const int nit = (nsb + 3) / 4;
    for (int it = 0; it < nit; it++) {
        const int  sb   = 4 * it + g;
        const bool act  = sb < nsb;
        const int  sbc  = act ? sb : nsb - 1;
        const int  nblk = min(nsb - 4 * it, 4);

        int m[R];
        uint4 v0[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int off = mis[r] + 4 * BS * it;
            m[r] = off & 15;
            const int nu4 = (m[r] + BS * nblk + 15) >> 4;
            v0[r] = lane < nu4 ? __ldg(gp[r] + (off >> 4) + lane) : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (lane < STAGE_U4) {
                stage[warp][r][lane] = v0[r];
            }
        }
        __syncwarp();

        // per-column activation scales and -4 offsets, independent of the row
        float rs[NC][4], so[NC][4];
#pragma unroll
        for (int cc = 0; cc < NC; cc++) {
            const float2 * ysc = ys + (int64_t) cc * (K / 32);
            const float4 rs01 = __ldg((const float4 *) (ysc + sbc * 8 + 4 * n));
            const float4 rs23 = __ldg((const float4 *) (ysc + sbc * 8 + 4 * n + 2));
            rs[cc][0] = rs01.x; rs[cc][1] = rs01.z; rs[cc][2] = rs23.x; rs[cc][3] = rs23.z;
            const float * sp = s16 + (int64_t) cc * (K / 16) + sbc * 16 + 8 * n + i0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                so[cc][j] = __ldg(sp + 2 * j) * offm;
            }
        }

#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint8_t  * bb  = (const uint8_t *) &stage[warp][r][0] + m[r] + BS * g;
            const uint32_t * w   = (const uint32_t *) ((uintptr_t) bb & ~(uintptr_t) 3);
            const uint32_t   shf = ((uintptr_t) bb & 3) * 8;
            auto ld4 = [&](const int byteoff) {
                const uint32_t * q = w + byteoff / 4;
                return __funnelshift_r(q[0], q[1], shf);
            };
            const uint32_t QS[2] = {ld4(32 + 32 * n + 8 * c), ld4(32 + 32 * n + 8 * c + 4)};
            const uint32_t HM[2] = {ld4(8 * c), ld4(8 * c + 4)};
            const uint32_t A = ld4(96), B = ld4(100), C = ld4(104);
            const uint32_t DW = ld4(108);
            const uint32_t W0 = ((A >> sA) & 0x0F0F0F0Fu) | (((C >> sC0) & 0x03030303u) << 4);
            const uint32_t W1 = ((B >> sA) & 0x0F0F0F0Fu) | (((C >> sC1) & 0x03030303u) << 4);

            uint32_t wv[4][4]; // [run j][2k + half]
#pragma unroll
            for (int k = 0; k < 2; k++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t v = ((QS[k] >> (2 * j)) & 0x03030303u) | (((HM[k] >> (4 * n + j)) & 0x01010101u) << 2);
                    wv[j][2*k + 0] = __byte_perm(v, 0, 0x4140);
                    wv[j][2*k + 1] = __byte_perm(v, 0, 0x4342);
                }
            }
            const float sc[4] = {gp100_byte_f(W0, i0) - 32.0f, gp100_byte_f(W0, i0 + 2) - 32.0f,
                                 gp100_byte_f(W1, i0) - 32.0f, gp100_byte_f(W1, i0 + 2) - 32.0f};
            const float dd = __half2float(__ushort_as_half((unsigned short) (DW & 0xFFFF)));
#pragma unroll
            for (int cc = 0; cc < NC; cc++) {
                const uint4 * yp = (const uint4 *) (y + (int64_t) cc * K + sbc * 256 + 128 * n + 8 * c);
                const uint4 yv[4] = {__ldg(yp), __ldg(yp + 4), __ldg(yp + 8), __ldg(yp + 12)};
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const half2 * Y = (const half2 *) &yv[j];
                    half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        acc = __hfma2(*(const half2 *) &wv[j][i], Y[i], acc);
                    }
                    s = fmaf(sc[j], fmaf(__half2float(__hadd(__low2half(acc), __high2half(acc))), rs[cc][j], so[cc][j]), s);
                }
                if (act) {
                    sum[r][cc] = fmaf(dd, s, sum[r][cc]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int cc = 0; cc < NC; cc++) {
            float v = sum[r][cc];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0 && row0 + r < nrows) {
                dst[(int64_t) cc * stride_dst + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vW, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Columns of a matmul operand: its rows, or its ne[2] when it is [K, 1, n] (one token per sequence), else 0.
static int64_t gp100_ncols(const ggml_tensor * t) {
    return t->ne[2] == 1 ? t->ne[1] : (t->ne[1] == 1 ? t->ne[2] : 0);
}
static int64_t gp100_col_stride(const ggml_tensor * t) {
    return (t->ne[2] == 1 ? t->nb[1] : t->nb[2]) / sizeof(float);
}

bool ggml_cuda_gp100_mmvq_supported(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (cc != GGML_CUDA_CC_PASCAL || GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
        return false;
    }
    static const bool disabled = getenv("GGML_CUDA_GP100_MMVQ_DISABLE") != nullptr;
    if (disabled) {
        return false;
    }
    // MUL_MAT_ID (MoE): every (expert slot, token) pair is one activation column against one expert matrix, so
    // the kernels take it as grid.y (slot) times grid.z (token) and rebase their pointers from ids there (see
    // gp100_pair_rebase). A MoE layer issues it in two forms: the down projection gives every slot its own
    // activation column, gate/up hand the token's single vector to all of its slots.
    const bool moe = ids != nullptr;
    if (moe && fusion) {
        return false;
    }
    // Isolates the expert path for A/B: without it the MoE matmuls fall back to the generic kernel while every
    // other matmul keeps the GP100 one, which is the only way to price this path on a MoE model.
    static const bool no_moe = getenv("GGML_CUDA_GP100_NO_MOE") != nullptr;
    if (moe && no_moe) {
        return false;
    }
    // Isolates this one type: GGML_CUDA_GP100_MMVQ_DISABLE=1 would also drop Q6_K/Q8_0/Q4_K
    // to the generic path, which makes A/B measurements of IQ3_S alone impossible.
    if (src0->type == GGML_TYPE_IQ3_S && getenv("GGML_CUDA_GP100_IQ3S_DISABLE") != nullptr) {
        return false;
    }
    // Both IQ3 types keep the fused pair on the generic path. The fusion doubles the accumulator set
    // (RT = 2R) and shares one activation read between ffn_up and ffn_gate, but measured on the 27B mix it
    // is a wash at its best: 18.83 tok/s fused against 18.88 with the two matvecs separate, because the
    // bottleneck here is the weight stream, not the activation stream. Not worth an unverified path.
    if (fusion && (src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ3_XXS)) {
        return false;
    }
    if (src0->type == GGML_TYPE_IQ3_XXS) {
        if (getenv("GGML_CUDA_GP100_IQ3XXS_DISABLE") != nullptr) {
            return false;
        }
        // Measured in situ on the 27B IQ3_S mix: the kernel ties the generic int8 path on plain
        // matvecs (+0.5%) but loses 1.5% with the swiglu fusion, where RT = 2R doubles the
        // accumulator set. Keep it off the fused pair until the GLU path is retuned.
        if (fusion) {
            return false;
        }
    }
    if (fusion) { // only the plain gate/up/SWIGLU fusion: no biases, no scales
        if (!fusion->gate || fusion->x_bias || fusion->gate_bias || fusion->x_scale || fusion->gate_scale ||
            fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->gate->type != src0->type ||
            fusion->gate->nb[1] != src0->nb[1] || (uintptr_t) fusion->gate->data % 16 != 0) {
            return false;
        }
    }
    switch (src0->type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
            break;
        default:
            return false;
    }
    // 1 column (all types above), or 2..8 columns (Q4_K/Q5_K/IQ4_XS/Q3_K/Q6_K/Q8_0 and both IQ3 types, no
    // fusion). The columns are either src1's rows (ne[1]) or, for one-token-per-sequence batches shaped
    // [K, 1, n_seqs] (qwen35's ssm_out), its ne[2].
    // columns the activation cache must hold: one per (slot, token), or one per token when the slots share it
    const int64_t ncols = moe ? src1->ne[1] * src1->ne[2] : gp100_ncols(src1);
    // Multi-column (2..8) only appears when a draft adds columns to a decode batch (speculative verify).
    // 2026-11-09: the earlier "garbage output" was seen with --spec-type ngram-simple, whose rejection path
    // never rolls back correctly on this hybrid (ssm) model -- it emits garbage with every backend, so it was
    // never evidence against these kernels. Re-tested with DSpine (--spec-type draft-dflash) at the model's
    // real shapes (k = 5120, n = 1..8, plus every (nrows, K) pair the 27B mix uses): all cases match the CPU
    // reference in test-backend-ops and the whole generation stays coherent.
    // On the 27B IQ3_S mix at n_max 5 (73-token prompt, 320 tokens out): 18.87 tok/s with one column,
    // 26.05 with the draft and columns still on the generic path, 36.13 with these kernels (2.46x over the
    // 14.66 tok/s integer path). Still opt-in because it changes the last bits, and only a caller that
    // verifies several tokens at once can use it: set GGML_CUDA_GP100_NC=1 (presence-only check, do not set
    // it to 0 expecting it off).
    static const bool nc_opt_in = getenv("GGML_CUDA_GP100_NC") != nullptr;
    if (!moe && ncols > 1 && !nc_opt_in) {
        return false;
    }
    if (ncols < 1 || (!moe && (ncols > 8 || gp100_ncols(dst) != ncols || src1->ne[1] != dst->ne[1])) ||
        (moe && ncols > 64) ||
        (!moe && ncols > 1 && (fusion ||
                               (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K &&
                                  src0->type != GGML_TYPE_IQ4_XS && src0->type != GGML_TYPE_IQ3_S &&
                                  src0->type != GGML_TYPE_IQ3_XXS &&
                                  src0->type != GGML_TYPE_Q3_K && !(src0->type == GGML_TYPE_Q8_0 && src0->ne[1] >= 2048) &&
                                  !(src0->type == GGML_TYPE_Q6_K && ncols <= 4))))) {
        // Multi-column Q8_0 takes large matrices only: at 4 columns it wins there (0.178 -> 0.085 ms) but the
        // 24-row alpha/beta projections run only 24 blocks and lose (8 -> 10 us each). IQ4_NL stays generic: its
        // multi-column kernel measured 0.96 vs 0.85 ms at 4 columns with 1, 2 or 4 rows per warp.
        // Q6_K's multi-column kernel (gp100_mmvq_q6_K_nc) loses to the generic int8 path at 5 columns in situ
        // (7.68 vs 6.18 ms per MTP verify batch) but wins at 4 concurrent sequences (batched decode 73.1 -> 74.2
        // tok/s aggregate), so it takes 2..4 columns only
        return false;
    }
    // IQ4_NL/Q8_0 are staged 32 blocks at a time, so they only need K % 32; every other type walks K in
    // super-blocks. (Unaligned row strides still lock the b32 types out: their staging reads the row base as
    // uint4, so nb[1] % 16 == 0 stays required below.)
    const bool b32_only = src0->type == GGML_TYPE_IQ4_NL || src0->type == GGML_TYPE_Q8_0;
    if (src0->ne[0] % (b32_only ? QK8_0 : QK_K) != 0) {
        return false;
    }
    if (moe) {
        // One index (slot-major, token-minor) drives the activation column, the output column and ids, so all
        // three have to be dense in that order. src1 is either one column per slot or a single broadcast column.
        if (src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]) {
            return false;
        }
        if (dst->ne[1] != ids->ne[0] || dst->ne[2] != ids->ne[1]) {
            return false;
        }
        if (src1->nb[1] != src0->ne[0] * (int64_t) sizeof(float) ||
            src1->nb[2] != src1->ne[1] * src1->nb[1] ||
            dst->nb[1]  != dst->ne[0] * (int64_t) sizeof(float) ||
            dst->nb[2]  != dst->ne[1] * dst->nb[1] ||
            ids->nb[1] % (int64_t) sizeof(int32_t) != 0) {
            return false;
        }
        if (src0->ne[2] < 2 || src0->nb[2] % 16 != 0) {
            return false;
        }
    } else if (src0->ne[2] != 1) {
        return false;
    }
    const bool ok = src1->ne[3] == 1 && dst->nb[0] == sizeof(float) &&
           src0->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
           src0->nb[1] % 2 == 0 && (uintptr_t) src0->data % 16 == 0 &&
           // Q6_K/Q3_K/IQ3_S/IQ3_XXS stage their blocks in shared memory; Q2_K reads qs with 32-bit loads and
           // scales with 16-bit ones, so it only needs 4-byte alignment (an 84-byte block gives nb[1] = 168 at K = 512)
           (src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_IQ3_S ||
            src0->type == GGML_TYPE_IQ3_XXS || src0->type == GGML_TYPE_Q2_K || src0->nb[1] % 16 == 0) &&
           (src0->type != GGML_TYPE_Q2_K || src0->nb[1] % 4 == 0);
    // GGML_CUDA_GP100_DIAG=1: why a MUL_MAT_ID did or did not take this path, once per distinct shape
    static const bool diag = getenv("GGML_CUDA_GP100_DIAG") != nullptr;
    if (diag && moe) {
        fprintf(stderr, "[gp100diag] moe ok=%d type=%2d K=%4lld M=%5lld slots=%2lld tok=%2lld ncols=%2lld "
                        "s0nb1=%6zu s0nb2=%8zu s1nb1=%6zu dstnb1=%5zu idsnb1=%4zu idsne0=%2lld fusion=%d\n",
                (int) ok, (int) src0->type, (long long) src0->ne[0], (long long) src0->ne[1],
                (long long) src1->ne[1], (long long) src1->ne[2], (long long) ncols,
                src0->nb[1], src0->nb[2], src1->nb[1], dst->nb[1], ids->nb[1], (long long) ids->ne[0],
                (int) (fusion != nullptr));
    }
    return ok;
           // Tiny Q8_0 matrices (ssm_alpha/beta, 24 rows/card) used to go back to the generic kernel, which won
           // 5.7 vs 10.1 us in situ because one warp per row walks all 160 weight blocks of K serially. The
           // K-split kernel splits that walk over the block's four warps, so they are admitted again.
}

// MUL_MAT_ID launch geometry. ids == nullptr means a plain matmul: one column, grid.y == 1, no rebasing.
struct gp100_id_launch {
    const int32_t * ids          = nullptr;
    int64_t         stride_xpert = 0; // bytes between expert matrices (src0->nb[2])
    int64_t         stride_y     = 0; // halfs between activation columns (K)
    int64_t         stride_dst   = 0; // floats between output columns (dst->ne[0])
    int             n_slots      = 1; // grid.y: expert slots per token (ids->ne[0])
    int             n_tok        = 1; // grid.z: tokens (ids->ne[1])
    int             ybc          = 0; // 1 when the slots of a token share one activation column
    int             stride_ids   = 0; // ids elements between tokens (ids->nb[1] / sizeof(int32_t))
};

typedef void (*gp100_mmvq_kernel_t)(const void *, const void *, const half *, const float2 *, const float *, float *, int, int, int,
                                    const int32_t *, int64_t, int64_t, int64_t, int, int, int, int);

template <gp100_mmvq_kernel_t kernel, int R, int KS = 1>
static void gp100_launch(const void * W, const void * G, const half * y, const float2 * ys, const float * s16, float * dst,
                         const int nrows, const int nsb, const int stride_row, const int n32,
                         const gp100_id_launch & id, cudaStream_t stream) {
    constexpr int RPB    = (4 / KS) * R;           // rows per block: 4 warps, KS of them sharing a row group
    const int     nblocks = (nrows + RPB - 1) / RPB;
    kernel<<<dim3(nblocks, id.n_slots, id.n_tok), 128, 0, stream>>>(W, G, y, ys, s16, dst, nrows, nsb, stride_row,
                                                         id.ids, id.stride_xpert, id.stride_y, id.stride_dst, n32,
                                                         id.n_slots, id.ybc, id.stride_ids);
}

// rows per warp: 8 keeps the most loads in flight, but small matrices need more warps than that gives
// (Q5_K 512 rows: R=2 11.7 us, R=8 19.8 us); Q6_K is best at 4 (380 vs 370 at 2, 8704x5120).
// With GLU each warp row does two matrices, so R is halved to keep the same register budget.
// Rows per warp. Defaults: 8 keeps the most loads in flight on big matrices, small ones need more warps
// (Q5_K 512 rows: R=2 11.7 us, R=8 19.8 us); Q6_K is best at 4. With GLU each warp row does two matrices,
// so R is halved. GP100_RMAP="nrows:R,..." overrides R for matrices with exactly nrows rows (Q6_K: R <= 4),
// so shapes can be tuned in situ without rebuilding.
static int gp100_rows_override(const int nrows, const bool glu) {
    static std::vector<std::pair<int, int>> rmap = [] {
        std::vector<std::pair<int, int>> m;
        if (const char * e = getenv("GP100_RMAP")) {
            int n, r, used;
            while (sscanf(e, "%d:%d%n", &n, &r, &used) == 2) {
                m.emplace_back(n, r);
                e += used;
                if (*e != ',') break;
                e++;
            }
        }
        return m;
    }();
    for (const auto & [n, r] : rmap) {
        if (n == nrows) {
            return glu ? std::max(1, r / 2) : r;
        }
    }
    return 0;
}

template <bool GLU, int R>
static void gp100_dispatch_r(const ggml_type type, const void * W, const void * G, const half * y, const float2 * ys,
                             const float * s16, float * dst, const int nrows, const int nsb, const int stride_row,
                             const int n32, const gp100_id_launch & id, cudaStream_t stream) {
#define GP100_LAUNCH(K) gp100_launch<K, R>(W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream)
// Short matrices take the K-split kernel: four warps share a row group instead of taking a row each, which
// quarters the serial walk over K. 2048 is the same row count at which R already drops to 1.
// The b32 kernels walk K in 32-blocks, not super-blocks, so they take n32 as their block count: that is what
// lets gpt-oss (K = 2880) through, where K % QK_K != 0.
#define GP100_LAUNCH_B32(T)                                                                                   \
    do {                                                                                                      \
        if (nrows < 2048) {                                                                                   \
            gp100_launch<(gp100_mmvq_b32<T, R, GLU, 4>), R, 4>(W, G, y, ys, s16, dst, nrows, n32, stride_row, n32, id, stream); \
        } else {                                                                                              \
            gp100_launch<(gp100_mmvq_b32<T, R, GLU, 1>), R>(W, G, y, ys, s16, dst, nrows, n32, stride_row, n32, id, stream);    \
        }                                                                                                     \
    } while (0)
    switch (type) {
        case GGML_TYPE_Q4_K:   GP100_LAUNCH((gp100_mmvq_kq<GGML_TYPE_Q4_K, R, GLU>)); break;
        case GGML_TYPE_Q5_K:   GP100_LAUNCH((gp100_mmvq_kq<GGML_TYPE_Q5_K, R, GLU>)); break;
        case GGML_TYPE_IQ4_XS: GP100_LAUNCH((gp100_mmvq_iq4_xs<R, GLU>));             break;
        case GGML_TYPE_Q8_0:
            if constexpr (R <= 4) {
                GP100_LAUNCH_B32(GGML_TYPE_Q8_0);
                break;
            }
            GGML_ABORT("Q8_0 supports at most 4 rows per warp");
#if 0 // PTQ1_0 excluded
        case GGML_TYPE_PTQ1_0:
            if constexpr (R <= (GLU ? 2 : 16)) {
                GP100_LAUNCH((gp100_mmvq_ptq1_0<R, GLU>));
                break;
            }
            GGML_ABORT("PTQ1_0 supports at most 16 rows per warp, 2 with GLU");
#endif
        case GGML_TYPE_Q2_K:
            // 32 quant bytes per thread against 48 for the kq kernels, so eight rows per warp still fit
            if constexpr (R <= (GLU ? 4 : 8)) {
                GP100_LAUNCH((gp100_mmvq_q2_K<R, GLU>));
                break;
            }
            GGML_ABORT("Q2_K supports at most 8 rows per warp, 4 with GLU");
        case GGML_TYPE_Q3_K:
            if constexpr (R <= 8) {
                GP100_LAUNCH((gp100_mmvq_q3_K<R, GLU>));
                break;
            }
            GGML_ABORT("Q3_K supports at most 8 rows per warp");
        case GGML_TYPE_IQ3_S:
            // measured on P100 at 8192x8192: R=2 191, R=3 203, R=4 183 GB/s; 8 rows spills
            if constexpr (R <= (GLU ? 2 : 4)) {
                GP100_LAUNCH((gp100_mmvq_iq3_s<R, GLU>));
                break;
            }
            GGML_ABORT("IQ3_S supports at most 4 rows per warp, 2 with GLU");
        case GGML_TYPE_IQ3_XXS:
            // measured on P100 at 8192x8192: R=2 206.6, R=4 168.6 GB/s
            if constexpr (R <= (GLU ? 2 : 4)) {
                GP100_LAUNCH((gp100_mmvq_iq3_xxs<R, GLU>));
                break;
            }
            GGML_ABORT("IQ3_XXS supports at most 4 rows per warp, 2 with GLU");
        case GGML_TYPE_IQ4_NL:
            if constexpr (R <= 4) {
                GP100_LAUNCH_B32(GGML_TYPE_IQ4_NL);
                break;
            }
            GGML_ABORT("IQ4_NL supports at most 4 rows per warp");
        case GGML_TYPE_Q6_K:
            // Must match the clamp in gp100_dispatch: only the non-GLU output head takes 8 rows. With GLU
            // the staging doubles, and <8, true> would need 4 x 16 x 896 = 57344 B of static shared memory,
            // over sm_60's 48 KB. CUDA 12.4's ptxas emits it anyway (it is never launched); 12.8 rejects it.
            if constexpr (R <= (GLU ? 2 : 8)) {
                GP100_LAUNCH((gp100_mmvq_q6_K<R, GLU>));
                break;
            }
            GGML_ABORT("Q6_K supports at most 8 rows per warp, 2 with GLU");
        default: GGML_ABORT("unsupported type for the GP100 MMVQ path");
    }
#undef GP100_LAUNCH
#undef GP100_LAUNCH_B32
}

template <bool GLU>
static void gp100_dispatch(const ggml_type type, const void * W, const void * G, const half * y, const float2 * ys,
                           const float * s16, float * dst, const int nrows, const int nsb, const int stride_row,
                           const int n32, const gp100_id_launch & id, cudaStream_t stream) {
    int R = gp100_rows_override(nrows, GLU);
    // The output head is the one Q6_K shape that wants 8 rows per warp rather than 4. It is 124160 rows,
    // and -sm tensor halves its K to 2560, so each row is only 10 super-blocks: the per-row setup and
    // epilogue dominate a very short inner loop, and it measures ~237 GB/s against ~466 for every other
    // Q6_K tensor. Eight rows amortise that. 4 warps x 8 rows x 896 B = 28672 B of staging still fits two
    // blocks per SM on GP100's 64 KB. Body tensors keep 4: they have long rows and want the occupancy.
    // Q3_K takes the same rule: the MTP draft head's output projection is Q3_K at 124160 rows x K 2560 and ran
    // ~136 GB/s at 4 rows per warp (staging is 480 B/row, so 8 rows fit in 15 KB).
    const bool q6k_wide = (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q3_K) && !GLU && nrows >= 32768;
    if (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q3_K) {
        R = std::min(R, q6k_wide ? 8 : (GLU ? 2 : 4)); // shared staging: 4 warps x RT rows x 896 (Q6_K) or 1088 (Q8_0) bytes
    }
    if (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
        R = std::min(R, GLU ? 2 : 4); // 203 GB/s at R=3 on P100; beyond 4 the register budget spills
    }
    if (type == GGML_TYPE_Q2_K) {
        R = std::min(R, GLU ? 4 : 8);
    }
    if (R == 0) {
        if (q6k_wide) {
            R = 8;
        } else if (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q3_K) {
            R = GLU ? 2 : 4;
        } else if (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
            R = 2; // 2 beats 4 in situ: R=4 measured 182.7 GB/s standalone vs 191 at R=2
        } else if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
            R = nrows < 2048 ? 1 : (GLU ? 2 : 4);
        } else {
            R = nrows < 2048 ? (GLU ? 1 : 2) : (GLU ? 4 : 8);
        }
    }
#if 0 // PTQ1_0 excluded
    if (type == GGML_TYPE_PTQ1_0) {
        static const int ptq_r = [] {
            const char * e = getenv("GGML_CUDA_GP100_PTQ_R");
            return e ? atoi(e) : 0;
        }();
        R = std::min(R, GLU ? 2 : 16);
        if (ptq_r > 0 && !GLU) {
            R = std::min(ptq_r, 16);
        }
    }
#endif
    switch (R) {
        case 1:  gp100_dispatch_r<GLU, 1>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
        case 2:  gp100_dispatch_r<GLU, 2>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
        case 4:  gp100_dispatch_r<GLU, 4>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
        case 6:  gp100_dispatch_r<GLU, 6>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
        case 16: gp100_dispatch_r<GLU, 16>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
        default: gp100_dispatch_r<GLU, 8>(type, W, G, y, ys, s16, dst, nrows, nsb, stride_row, n32, id, stream); break;
    }
}

// 2..8 columns: 4 rows per warp, R x NC accumulators per thread. In situ at 4 columns (batched decode, tok/s
// aggregate): R=1 42.8, R=2 62.3, R=4 74.3, R=8 66.9.
template <ggml_type type, int NC>
static void gp100_launch_nc(const void * W, const half * y, const float2 * ys, const float * s16, float * dst, const int nrows, const int nsb,
                            const int stride_row, const int stride_dst, cudaStream_t stream) {
    constexpr int R = 4;
    const int nblocks = (nrows + 4*R - 1) / (4*R);
    if constexpr (type == GGML_TYPE_Q6_K) {
        gp100_mmvq_q6_K_nc<R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        gp100_mmvq_q3_K_nc<R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // short matrices split K over the block's 4 warps (as gp100_mmvq_b32 does), long ones take 2 rows per warp
        // large matrices only (the support gate keeps short ones on the generic kernel); 4 rows per warp
        // (in situ at 4 columns: R=1 0.133, R=2 0.122, R=4 0.085 ms)
        // the single-column b32 kernels take 32-block counts (K % QK_K may be non-zero); the multi-column path
        // only runs at K % QK_K == 0, so the super-block count is converted back here
        gp100_mmvq_b32_nc<type, 4, 1, NC><<<(nrows + 15) / 16, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb * (QK_K / QK8_0), stride_row, stride_dst);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        gp100_mmvq_iq4_xs_nc<R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        // R is the rows-per-warp knob and it trades two things against each other: a larger R reuses the
        // activation reads over more rows but raises the live register count, and at NC = 6 the R = 4 build
        // already needs 202 registers, i.e. 2 blocks per SM (8 warps), which is not enough to cover memory
        // latency. GP100_NC_R overrides it in situ: 1, 2, 4.
        static const int nc_r = [] { const char * e = getenv("GP100_NC_R"); return e ? atoi(e) : 4; }();
        if (nc_r == 1) {
            gp100_mmvq_iq3_s_nc<1, NC><<<(nrows + 3) / 4, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
        } else if (nc_r == 2) {
            gp100_mmvq_iq3_s_nc<2, NC><<<(nrows + 7) / 8, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
        } else if (nc_r == 8) {
            gp100_mmvq_iq3_s_nc<8, NC><<<(nrows + 31) / 32, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
        } else {
            gp100_mmvq_iq3_s_nc<R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
        }
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        gp100_mmvq_iq3_xxs_nc<R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    } else {
        gp100_mmvq_kq_nc<type, R, NC><<<nblocks, 128, 0, stream>>>(W, y, ys, dst, nrows, nsb, stride_row, stride_dst);
    }
}

template <ggml_type type>
static void gp100_dispatch_nc_t(const void * W, const half * y, const float2 * ys, const float * s16, float * dst, const int nrows, const int nsb,
                                const int stride_row, const int ncols, const int stride_dst, cudaStream_t stream) {
    switch (ncols) {
        case 2: gp100_launch_nc<type, 2>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 3: gp100_launch_nc<type, 3>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 4: gp100_launch_nc<type, 4>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 5: gp100_launch_nc<type, 5>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 6: gp100_launch_nc<type, 6>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 7: gp100_launch_nc<type, 7>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        case 8: gp100_launch_nc<type, 8>(W, y, ys, s16, dst, nrows, nsb, stride_row, stride_dst, stream); break;
        default: GGML_ABORT("GP100 multi-column MMVQ supports 2..8 columns");
    }
}

static void gp100_dispatch_nc(const ggml_type type, const void * W, const half * y, const float2 * ys, const float * s16, float * dst, const int nrows,
                              const int nsb, const int stride_row, const int ncols, const int stride_dst, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q4_K: gp100_dispatch_nc_t<GGML_TYPE_Q4_K>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_Q5_K: gp100_dispatch_nc_t<GGML_TYPE_Q5_K>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_IQ4_XS: gp100_dispatch_nc_t<GGML_TYPE_IQ4_XS>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_Q6_K: gp100_dispatch_nc_t<GGML_TYPE_Q6_K>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_Q3_K: gp100_dispatch_nc_t<GGML_TYPE_Q3_K>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_Q8_0: gp100_dispatch_nc_t<GGML_TYPE_Q8_0>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_IQ3_S: gp100_dispatch_nc_t<GGML_TYPE_IQ3_S>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        case GGML_TYPE_IQ3_XXS: gp100_dispatch_nc_t<GGML_TYPE_IQ3_XXS>(W, y, ys, s16, dst, nrows, nsb, stride_row, ncols, stride_dst, stream); break;
        default: GGML_ABORT("no GP100 multi-column kernel for this type");
    }
}

// Several matmuls in a row often read the same activations (qkv, ssm_alpha and ssm_beta; the four attention
// projections; ffn gate and up when they are not fused). The converted activations are kept in a persistent
// per-(device, stream) buffer and reused when the next matmul has the same src1 tensor, data pointer and K
// within the same graph evaluation: between a tensor's producer and its last consumer its data cannot change.
struct gp100_act_cache {
    char *                buf   = nullptr;
    const ggml_tensor *   src1  = nullptr;
    const void *          data  = nullptr;
    int64_t               K     = 0;
    int64_t               ncols = 0;
    uint64_t              epoch = 0;
};
static constexpr int64_t GP100_ACT_CACHE_MAX_K = 8 * 17408; // up to 8 columns of the widest K in this model
static constexpr size_t  GP100_OFF_YS  = GGML_PAD(GP100_ACT_CACHE_MAX_K * sizeof(half), 256);
static constexpr size_t  GP100_OFF_S16 = GP100_OFF_YS + GGML_PAD(GP100_ACT_CACHE_MAX_K / 32 * sizeof(float2), 256);
static constexpr size_t  GP100_ACT_BYTES = GP100_OFF_S16 + GP100_ACT_CACHE_MAX_K / 16 * sizeof(float);

static gp100_act_cache & gp100_cache(ggml_backend_cuda_context & ctx) {
    static gp100_act_cache caches[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];
    gp100_act_cache & c = caches[ctx.device][ctx.curr_stream_no];
    if (c.buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&c.buf, GP100_ACT_BYTES));
    }
    return c;
}

static bool gp100_cache_enabled() {
    static const bool no_cache = getenv("GGML_CUDA_GP100_NO_ACT_CACHE") != nullptr;
    return !no_cache;
}

// Allocate the GPU-busy stamp buffer on first use and flush it once it is nearly full. The CUDA
// driver is already torn down by the time an atexit handler runs, so the dump has to happen from
// inside a launch: a 4-byte async copy into pinned memory tells the host how far the counter has
// got without stalling the stream.
static unsigned long long * gp100_ts_dev    = nullptr;
static unsigned int         gp100_ts_used   = 0;
static void *               gp100_ts_n_addr = nullptr; // device address of gp100_ts_n
static volatile int *       gp100_ts_n_host = nullptr; // pinned, written by an async copy
static bool                 gp100_ts_dumped = false;

static void gp100_ts_flush() {
    if (gp100_ts_dev == nullptr || gp100_ts_dumped) {
        return;
    }
    gp100_ts_dumped = true;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[gp100ts] sync failed while flushing\n");
        return;
    }
    unsigned int n = 0;
    cudaError_t e = cudaMemcpyFromSymbol(&n, gp100_ts_n, sizeof(n));
    if (e != cudaSuccess) {
        fprintf(stderr, "[gp100ts] cannot read the CTA counter: %s\n", cudaGetErrorString(e));
        return;
    }
    if (n > gp100_ts_used) {
        n = gp100_ts_used;
    }
    std::vector<unsigned long long> h((size_t) 2 * n);
    e = cudaMemcpy(h.data(), gp100_ts_dev, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gp100ts] cannot read the buffer: %s\n", cudaGetErrorString(e));
        return;
    }
    const char * path = getenv("GGML_CUDA_GP100_TS_OUT");
    if (path == nullptr) {
        path = "D:/bench/gp100_ts.bin";
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "[gp100ts] cannot open %s\n", path);
        return;
    }
    fwrite(h.data(), sizeof(unsigned long long), h.size(), f);
    fclose(f);
    fprintf(stderr, "[gp100ts] %u CTA stamps -> %s\n", n, path);
}

static void gp100_ts_setup() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    if (getenv("GGML_CUDA_GP100_TS") == nullptr) {
        return;
    }
    const unsigned int cap = 1u << 19; // 524k CTAs = 8 MiB, ~7 decode tokens
    unsigned long long * p = nullptr;
    if (cudaMalloc(&p, (size_t) 2 * cap * sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "[gp100ts] cudaMalloc failed\n");
        return;
    }
    cudaMemset(p, 0, (size_t) 2 * cap * sizeof(unsigned long long));
    cudaMemcpyToSymbol(gp100_ts_buf, &p, sizeof(p));
    cudaMemcpyToSymbol(gp100_ts_cap, &cap, sizeof(cap));
    cudaGetSymbolAddress(&gp100_ts_n_addr, gp100_ts_n);
    cudaHostAlloc((void **) &gp100_ts_n_host, sizeof(int), cudaHostAllocDefault);
    *gp100_ts_n_host = 0;
    gp100_ts_dev  = p;
    gp100_ts_used = cap;
    fprintf(stderr, "[gp100ts] armed with %u CTA slots\n", cap);
}

// Called on every matmul: cheap enough to run unconditionally, and it stops sampling once the
// buffer is written out.
static bool gp100_ts_armed = false;
static long gp100_ts_launches = 0;

static void gp100_ts_tick(cudaStream_t stream) {
    if (!gp100_ts_armed) {
        gp100_ts_armed = gp100_ts_dev != nullptr;
        if (!gp100_ts_armed) {
            return;
        }
    }
    if (gp100_ts_dumped || (gp100_ts_launches++ & 63) != 0) {
        return;
    }
    cudaMemcpyAsync((void *) gp100_ts_n_host, gp100_ts_n_addr, sizeof(int), cudaMemcpyDeviceToHost, stream);
    static const bool dbg = getenv("GGML_CUDA_GP100_TS_DEBUG") != nullptr;
    if (dbg) {
        fprintf(stderr, "[gp100ts] n=%d\n", *gp100_ts_n_host);
    }
    if (*gp100_ts_n_host > (int) gp100_ts_used - 70000) {
        gp100_ts_flush();
    }
}

// Optional per-matmul timing (GGML_CUDA_GP100_PROF=1): every launch is bracketed by two events and the stream
// is synchronized, so each number is that matmul's exclusive time but the run as a whole is slower. Use it to
// split a decode step into "matmul" against "everything else" (attention, ssm scan, norms, sampling), never to
// measure throughput.
static double      gp100_prof_ms[4] = {0, 0, 0, 0};
static long        gp100_prof_n [4] = {0, 0, 0, 0};
static cudaEvent_t gp100_prof_ev[2] = {nullptr, nullptr};

static void gp100_prof_report() {
    const char * name[4] = {"1 col", "nc   ", "glu  ", "moe  "};
    double tot = 0;
    long   cnt = 0;
    for (int i = 0; i < 4; i++) {
        tot += gp100_prof_ms[i];
        cnt += gp100_prof_n[i];
    }
    fprintf(stderr, "[gp100prof] total %.1f ms over %ld matmuls\n", tot, cnt);
    for (int i = 0; i < 4; i++) {
        if (gp100_prof_n[i] == 0) continue;
        fprintf(stderr, "[gp100prof]  %s: %8.1f ms  n=%6ld  %6.3f ms/call\n",
                name[i], gp100_prof_ms[i], gp100_prof_n[i], gp100_prof_ms[i] / gp100_prof_n[i]);
    }
}

void ggml_cuda_gp100_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    cudaStream_t stream = ctx.stream();
    gp100_ts_setup();
    gp100_ts_tick(stream);
    static const bool prof = getenv("GGML_CUDA_GP100_PROF") != nullptr;
    if (prof && gp100_prof_ev[0] == nullptr) {
        cudaEventCreate(&gp100_prof_ev[0]);
        cudaEventCreate(&gp100_prof_ev[1]);
        atexit(gp100_prof_report);
    }
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    // A MoE pair (expert slot, token) carries one activation column like a decode token does, so the whole
    // column machinery below (activation cache, R rows per warp, K-split) applies unchanged. The expert itself
    // is one grid step away: supported() has already checked that ids, src1 and dst are dense in that order.
    // gate/up of a MoE layer pass the token's single vector for all of its slots (ybc), the down projection
    // passes one column per slot - either way there is exactly one activation column per (slot, token).
    const int     ncols = ids ? (int) (src1->ne[1] * src1->ne[2]) : (int) gp100_ncols(src1);
    const int     nb32  = K / 32;
    const int64_t col_stride = ids ? (int64_t) (src1->nb[1] / sizeof(float)) : gp100_col_stride(src1);

    gp100_id_launch id;
    if (ids) {
        id.ids          = (const int32_t *) ids->data;
        id.stride_xpert = src0->nb[2];
        id.stride_y     = K;
        id.stride_dst   = dst->nb[1] / sizeof(float);
        id.n_slots      = (int) ids->ne[0];
        id.n_tok        = (int) ids->ne[1];
        id.ybc          = src1->ne[1] == 1 ? 1 : 0;
        id.stride_ids   = (int) (ids->nb[1] / sizeof(int32_t));
    }

    ggml_cuda_pool_alloc<char> tmp(ctx.pool());
    char * base;
    size_t off_ys  = GP100_OFF_YS;
    size_t off_s16 = GP100_OFF_S16;
    bool   hit = false;
    if (gp100_cache_enabled() && K * ncols <= GP100_ACT_CACHE_MAX_K) {
        gp100_act_cache & c = gp100_cache(ctx);
        hit = c.src1 == src1 && c.data == src1->data && c.K == K && c.ncols == ncols && c.epoch == 0;
        c.src1 = src1; c.data = src1->data; c.K = K; c.ncols = ncols; c.epoch = 0;
        base = c.buf;
    } else {
        // sized for THIS matmul: a fixed GP100_ACT_BYTES overflows once K * ncols > GP100_ACT_CACHE_MAX_K
        // (e.g. a DFlash drafter's fc, K = 25600, at 6+ columns)
        const int64_t n = std::max<int64_t>(K * ncols, GP100_ACT_CACHE_MAX_K);
        off_ys  = GGML_PAD(n * sizeof(half), 256);
        off_s16 = off_ys + GGML_PAD(n / 32 * sizeof(float2), 256);
        base = tmp.alloc(off_s16 + n / 16 * sizeof(float));
    }
    half   * y   = (half   *) base;
    float2 * ys  = (float2 *) (base + off_ys);
    float  * s16 = (float  *) (base + off_s16);
    if (!hit) {
        const int nblk = nb32 * ncols;
        gp100_prep_act<<<(nblk + 3) / 4, 128, 0, stream>>>((const float *) src1->data, y, ys, s16, nblk, nb32, col_stride);
    }

    const int nsb        = K / QK_K;
    const int stride_row = src0->nb[1] / ggml_type_size(src0->type);
    const int prof_slot  = fusion ? 2 : (ids ? 3 : (ncols > 1 ? 1 : 0));
    if (prof) {
        cudaEventRecord(gp100_prof_ev[0], stream);
    }
    if (ncols > 1 && !ids) {
        gp100_dispatch_nc(src0->type, src0->data, y, ys, s16, (float *) dst->data, nrows, nsb, stride_row, ncols, gp100_col_stride(dst), stream);
    } else if (fusion) {
        gp100_dispatch<true >(src0->type, src0->data, fusion->gate->data, y, ys, s16, (float *) dst->data, nrows, nsb, stride_row, nb32, id, stream);
    } else {
        gp100_dispatch<false>(src0->type, src0->data, nullptr,            y, ys, s16, (float *) dst->data, nrows, nsb, stride_row, nb32, id, stream);
    }
    if (prof) {
        cudaEventRecord(gp100_prof_ev[1], stream);
        cudaEventSynchronize(gp100_prof_ev[1]);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, gp100_prof_ev[0], gp100_prof_ev[1]);
        gp100_prof_ms[prof_slot] += ms;
        gp100_prof_n [prof_slot] += 1;
        // periodic dump instead of atexit: the server is usually killed, which skips atexit
        static long since = 0;
        if (++since >= 1024) {
            since = 0;
            gp100_prof_report();
        }
    }
}

// Fused RMS_NORM * weight for one row, which also produces the fp16 activations for the matmuls that consume
// it. The stock kernel runs the whole 5120-float row in ONE block (latency bound, ~9.4 us in situ); here every
// block recomputes sum(x^2) for the full row from L2 (20 KB) and then handles its own 32-element chunks.
// Writes the fp32 result the graph expects AND y/ys/s16 into the activation cache, keyed to the MUL node, so
// the following matmuls skip their own conversion.
static __global__ void gp100_rms_norm_mul_prep(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ out,
                                               half * __restrict__ y, float2 * __restrict__ ys, float * __restrict__ s16,
                                               const int K, const float eps, const int64_t x_stride) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    __shared__ float red[4];
    // row blockIdx.y (one per sequence in a batched decode): its own input row, output column and cache column
    x   += blockIdx.y * x_stride;
    out += blockIdx.y * K;
    y   += blockIdx.y * K;
    ys  += blockIdx.y * (K / 32);
    s16 += blockIdx.y * (K / 16);
    const float4 * x4 = (const float4 *) x;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) {
        const float4 v = __ldg(x4 + i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xFFFFFFFF, ss, o);
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        red[threadIdx.x / WARP_SIZE] = ss;
    }
    __syncthreads();
    ss = red[0] + red[1] + red[2] + red[3];
    const float scale = rsqrtf(ss / K + eps);

    const int b = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE; // 32-block
    const int l = threadIdx.x % WARP_SIZE;
    if (b >= K / 32) {
        return;
    }
    const float v = scale * x[b * 32 + l] * w[b * 32 + l];
    out[b * 32 + l] = v;

    float a = fabsf(v);
    float s = v;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
        s += __shfl_xor_sync(0xFFFFFFFF, s, o);
    }
    if (l % 16 == 0) {
        s16[2 * b + l / 16] = s;
    }
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, 16));
    s += __shfl_xor_sync(0xFFFFFFFF, s, 16);
    int e = 0;
    if (a > 0.0f) {
        frexpf(a, &e);
    }
    const int k = 15 - e;
    y[b * 32 + l] = __float2half_rn(ldexpf(v, k));
    if (l == 0) {
        ys[b] = make_float2(ldexpf(1.0f, 24 - k), s);
    }
#else
    GGML_UNUSED_VARS(x, w, out, y, ys, s16, K, eps, x_stride);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

bool ggml_cuda_gp100_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, const bool multi_row) {
    static const bool disabled = getenv("GGML_CUDA_GP100_NO_NORM_FUSION") != nullptr;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (disabled || cc != GGML_CUDA_CC_PASCAL || !gp100_cache_enabled()) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const int64_t K = x->ne[0];
    // one row, or (multi_row: the caller found a matmul reading the result) one row per sequence of a batched
    // decode, up to the 8 columns the matmuls take
    const int64_t nrows = ggml_nrows(x);
    if ((nrows > 1 && !multi_row) || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        nrows < 1 || nrows > 8 || ggml_nrows(w) != 1 || w->ne[0] != K || ggml_nrows(mul) != nrows || mul->ne[0] != K ||
        !ggml_is_contiguous_rows(x) || x->ne[3] != 1 || gp100_ncols(x) != nrows || !ggml_is_contiguous(w) || !ggml_is_contiguous(mul) ||
        gp100_ncols(mul) != nrows || K % QK_K != 0 || K * nrows > GP100_ACT_CACHE_MAX_K ||
        (uintptr_t) x->data % 16 != 0 || gp100_col_stride(x) % 4 != 0) {
        return false;
    }
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    gp100_act_cache & c = gp100_cache(ctx);
    const int nb32 = K / 32;
    gp100_rms_norm_mul_prep<<<dim3((nb32 + 3) / 4, nrows), 128, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul->data,
        (half *) c.buf, (float2 *) (c.buf + GP100_OFF_YS), (float *) (c.buf + GP100_OFF_S16), K, eps, gp100_col_stride(x));
    c.src1 = mul; c.data = mul->data; c.K = K; c.ncols = nrows; c.epoch = 0;
    return true;
}

// Gated RMS norm per head, silu(z) * (rms_norm(x) * w), for rows of D = blockDim.x (64/128/256) floats, which
// also fills the activation cache for the matmul that consumes it. One block per row, one warp per 32-block.
// Replaces rms_norm_f32<256, true> + unary_gated(silu, mul) + gp100_prep_act. The row sum is reduced exactly as
// block_reduce does it in the stock 256-thread kernel (per-warp xor butterfly, then a butterfly over the warp
// partials padded with zeros -- the stock kernel's extra warps contribute exact zeros), and every product is
// formed in the stock order, so the f32 output is bit-identical.
static __global__ void gp100_gated_norm_prep(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ z,
                                             float * __restrict__ out, half * __restrict__ y, float2 * __restrict__ ys,
                                             float * __restrict__ s16, const float eps) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    __shared__ float red[WARP_SIZE];
    const int D   = blockDim.x;
    const int col = threadIdx.x;
    const int i   = blockIdx.x * D + col;
    const int l   = col % WARP_SIZE;

    const float xi = x[i];
    float tmp = warp_reduce_sum(xi * xi);
    if (l == 0) {
        red[col / WARP_SIZE] = tmp;
    }
    __syncthreads();
    tmp = l < D / WARP_SIZE ? red[l] : 0.0f;
    tmp = warp_reduce_sum(tmp);
    const float scale = rsqrtf(tmp / D + eps);

    const float v = ggml_cuda_op_silu_single(z[i]) * (scale * xi * w[col]);
    out[i] = v;

    const int b = i / 32;
    float a = fabsf(v);
    float s = v;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
        s += __shfl_xor_sync(0xFFFFFFFF, s, o);
    }
    if (l % 16 == 0) {
        s16[2 * b + l / 16] = s;
    }
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, 16));
    s += __shfl_xor_sync(0xFFFFFFFF, s, 16);
    int e = 0;
    if (a > 0.0f) {
        frexpf(a, &e);
    }
    const int k = 15 - e;
    y[i] = __float2half_rn(ldexpf(v, k));
    if (l == 0) {
        ys[b] = make_float2(ldexpf(1.0f, 24 - k), s);
    }
#else
    GGML_UNUSED_VARS(x, w, z, out, y, ys, s16, eps);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

bool ggml_cuda_gp100_gated_norm_supported(const int64_t D, const int64_t K) {
    return gp100_cache_enabled() && (D == 64 || D == 128 || D == 256) && K % QK_K == 0 && K <= GP100_ACT_CACHE_MAX_K;
}

void ggml_cuda_gp100_gated_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z,
                                ggml_tensor * out, const ggml_tensor * key, const float eps) {
    const int64_t D = x->ne[0];
    const int64_t K = ggml_nelements(out);
    GGML_ASSERT(ggml_cuda_gp100_gated_norm_supported(D, K));
    gp100_act_cache & c = gp100_cache(ctx);
    gp100_gated_norm_prep<<<K / D, D, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (const float *) z->data,
        (float *) out->data, (half *) c.buf, (float2 *) (c.buf + GP100_OFF_YS), (float *) (c.buf + GP100_OFF_S16), eps);
    c.src1 = key; c.data = key->data; c.K = K; c.epoch = 0;
}
