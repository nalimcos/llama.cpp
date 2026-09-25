#pragma once

// Shared "wire" selection and quantized-wire kernels for cross-GPU all-reduce.
//
// A *wire* is the datatype actually shuffled over the interconnect during an
// all-reduce (or, for the internal AllReduce, the datatype staged to/from host
// memory).  The quantized wires (Q8_0, Q5_0, Q4_0) are block-quantized and can
// therefore only be summed by a code path we fully control: the internal
// AllReduce (2-GPU) or the N-GPU quantized butterfly in ggml-cuda.cu.  NCCL
// cannot express a block-quantized sum: the per-block scale differs both within
// a rank (each 32-element block has its own fp16 delta) and across ranks, so it
// cannot be factored out of a plain integer sum, and there is no nccl datatype
// that understands {fp16 d, int8 qs[32]}.  The NCCL ring is therefore limited
// to F32 / BF16 wires; when a quantized wire is selected the CUDA comm layer
// routes the reduction to the butterfly instead.
//
// Every all-reduce entry point (internal pipeline, NCCL comm path, quantized
// butterfly) shares one config instance so the byte-threshold ladder is
// consistent across the mesh.

#include "common.cuh"
#include "cpy-utils.cuh"
#include "dequantize.cuh"

#include <cstdint>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Wire config
// ---------------------------------------------------------------------------

// Byte thresholds selecting the on-wire datatype for an F32 reduction.  Most->
// least-aggressive on-wire narrowing:
//   F32 4 B/el (baseline) < BF16 2 B/el < Q8_0 34/32 ~= 1.06 < Q5_0 0.75 < Q4_0 0.5625
// A type is used only if its threshold is non-zero and the tensor's F32 byte
// size is at or above it; the most aggressive enabled+met type wins.
struct ggml_cuda_ar_wire_config {
    uint64_t bf16_threshold = 131072; // 128 KiB
    uint64_t q8_0_threshold = 0;      // disabled by default (numerics)
    uint64_t q5_0_threshold = 0;
    uint64_t q4_0_threshold = 0;
};

// Read an unsigned env var (strtoull), returning default_value when unset,
// empty, or unparseable.
static inline uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

// Parse all AR wire thresholds from the environment in one place.  Shared by
// the internal AllReduce pipeline init and the NCCL comm path.
static inline ggml_cuda_ar_wire_config ggml_cuda_ar_wire_config_get(void) {
    ggml_cuda_ar_wire_config cfg;
    cfg.bf16_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 131072);
    cfg.q8_0_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_Q8_0_THRESHOLD", 0);
    cfg.q5_0_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_Q5_0_THRESHOLD", 0);
    cfg.q4_0_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_Q4_0_THRESHOLD", 0);
    return cfg;
}

// Whether an F32 tensor with `ne` elements can travel over a quantized wire,
// i.e. whether its element count is rotatable by the Walsh-Hadamard transform.
// Implemented in fwht.cu (declared in fwht.cuh).
int ggml_cuda_ar_fwht_n(int64_t ne);

static inline bool ggml_cuda_ar_quant_eligible(int64_t ne) {
    return ggml_cuda_ar_fwht_n(ne) != 0;
}

// Pick the wire type for an F32 tensor of `nbytes` bytes.  `quant_eligible`
// must be true for a quantized wire to be selected (the caller computes it as
// rotatable ne).  Returns GGML_TYPE_F32, GGML_TYPE_BF16, or a block-quantized
// type.
static inline ggml_type ggml_cuda_ar_pick_wire(
        const ggml_cuda_ar_wire_config & cfg, bool quant_eligible, uint64_t nbytes) {
    if (quant_eligible) {
        if (cfg.q4_0_threshold > 0 && nbytes >= cfg.q4_0_threshold) return GGML_TYPE_Q4_0;
        if (cfg.q5_0_threshold > 0 && nbytes >= cfg.q5_0_threshold) return GGML_TYPE_Q5_0;
        if (cfg.q8_0_threshold > 0 && nbytes >= cfg.q8_0_threshold) return GGML_TYPE_Q8_0;
    }
    if (cfg.bf16_threshold > 0 && nbytes >= cfg.bf16_threshold) return GGML_TYPE_BF16;
    return GGML_TYPE_F32;
}

// ---------------------------------------------------------------------------
// Quantized-wire kernels (shared by the internal AllReduce and the NCCL
// N-GPU quantized butterfly).  These operate on flat contiguous buffers.
// ---------------------------------------------------------------------------

// Quantize a rotated F32 buffer (ne elements) blockwise into the wire type.
// One thread per block; reuses the block quantizers from cpy-utils.cuh.
template <int QK, typename block_t, void (*quantize)(const float *, block_t *)>
static __global__ void ggml_cuda_ar_quantize_kernel(
        const float * __restrict__ src,
        block_t     * __restrict__ dst,
        int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nblocks) {
        quantize(src + (int64_t) b * QK, dst + b);
    }
}

// Blockwise quantize an F32 buffer `src` (ne elements) into the on-wire
// buffer `dst` (ne/QK block_t entries) for `wire_type` (Q8_0/Q5_0/Q4_0).
static inline void ggml_cuda_ar_quantize(
        cudaStream_t stream, void * dst, const float * src, int64_t ne,
        ggml_type wire_type) {
    const int block_size = 256;
    switch (wire_type) {
        case GGML_TYPE_Q8_0: {
            const int nblocks = (int)(ne / QK8_0);
            const int n_blocks = (nblocks + block_size - 1) / block_size;
            ggml_cuda_ar_quantize_kernel<QK8_0, block_q8_0, quantize_f32_q8_0_block>
                <<<n_blocks, block_size, 0, stream>>>(
                    src, static_cast<block_q8_0 *>(dst), nblocks);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case GGML_TYPE_Q5_0: {
            const int nblocks = (int)(ne / QK5_0);
            const int n_blocks = (nblocks + block_size - 1) / block_size;
            ggml_cuda_ar_quantize_kernel<QK5_0, block_q5_0, quantize_f32_q5_0_block>
                <<<n_blocks, block_size, 0, stream>>>(
                    src, static_cast<block_q5_0 *>(dst), nblocks);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case GGML_TYPE_Q4_0: {
            const int nblocks = (int)(ne / QK4_0);
            const int n_blocks = (nblocks + block_size - 1) / block_size;
            ggml_cuda_ar_quantize_kernel<QK4_0, block_q4_0, quantize_f32_q4_0_block>
                <<<n_blocks, block_size, 0, stream>>>(
                    src, static_cast<block_q4_0 *>(dst), nblocks);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            GGML_ASSERT(false);
    }
}

// Quantized-wire add kernel.  `dst` is a *rotated* F32 accumulator (ne
// elements) and `src` is the peer's wire buffer of nblocks = ne/QK block_t
// entries.  Each thread owns one wire block.
//
// For equal precision between the two parties, both sides round their own
// rotated accumulator through the same wire before summing: the peer already
// quantized and transmitted its value (so we dequantize it back), and we
// re-quantize the *local* rotated accumulator with the same block quantizer
// and dequantize that too -- so both sides compute with identical per-element
// wire values.  An inactive shard was zeroed up-front, so its rotated value is
// exactly zero and re-quantizing it yields the same zero wire as the peer's.
// Since float addition is commutative, the order in which the two rounded
// terms are summed never matters for a single peer.
//
// `pairwise` selects the on-wire layout convention, matching dequantize.cuh:
//   * pairwise=false (Q4_0/Q5_0): dequant(ib, j) returns the two values stored
//     in byte j of qs[] (elements j and j + QK/2 of the block).
//   * pairwise=true  (Q8_0):       dequant(ib, 2*j) returns elements
//     2*j and 2*j + 1 (int8 quants are contiguously indexed).
template <typename block_t, int QK, bool pairwise,
          void (*dequant)(const void *, int64_t, int, float2 &),
          void (*quantize)(const float *, block_t *)>
static __global__ void ggml_cuda_ar_add_q_kernel(
        float            * __restrict__ dst,
        const block_t    * __restrict__ src,
        int               nblocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int b = tid; b < nblocks; b += nt) {
        const int off = b * QK;
        // Round the local (rotated) F32 accumulator through the same wire type
        // the peer used, so both sides sum identical per-element wire values.
        block_t local_q;
        quantize(dst + off, &local_q);
        float vals[QK];
#pragma unroll
        for (int j = 0; j < QK / 2; ++j) {
            float2 v_peer;
            dequant(src + b, 0, pairwise ? 2*j : j, v_peer);
            float2 v_local;
            dequant(&local_q, 0, pairwise ? 2*j : j, v_local);
            if (pairwise) {
                vals[2*j]     = v_peer.x + v_local.x;
                vals[2*j + 1] = v_peer.y + v_local.y;
            } else {
                vals[j]         = v_peer.x + v_local.x;
                vals[j + QK/2]  = v_peer.y + v_local.y;
            }
        }
#pragma unroll
        for (int j = 0; j < QK; ++j) {
            dst[off + j] = vals[j];
        }
    }
}

// Host wrapper: dequantize a peer's on-wire buffer `src` (nblocks = ne/QK
// block_t entries) and add it into `dst`, a rotated F32 accumulator (ne
// elements), re-quantizing the local accumulator through the same wire.
template <typename block_t, int QK, bool pairwise,
          void (*dequant)(const void *, int64_t, int, float2 &),
          void (*quantize)(const float *, block_t *)>
static inline void ggml_cuda_ar_add_q(
        cudaStream_t stream, float * dst, const void * src, int64_t ne) {
    const int nblocks = (int)(ne / QK); // ne is a multiple of QK
    const int block_size = 256;
    int n_blocks = (nblocks + block_size - 1) / block_size;
    if (n_blocks > 1024) {
        n_blocks = 1024;
    }
    ggml_cuda_ar_add_q_kernel<block_t, QK, pairwise, dequant, quantize>
        <<<n_blocks, block_size, 0, stream>>>(
            dst, static_cast<const block_t *>(src), nblocks);
    CUDA_CHECK(cudaGetLastError());
}

// Convenience dispatch: run the quantized add for a given wire type on
// `dst` (rotated F32 accumulator of ne elements) adding `src` (wire buffer).
static inline void ggml_cuda_ar_add_q_for_type(
        cudaStream_t stream, float * dst, const void * src, int64_t ne,
        ggml_type wire_type) {
    switch (wire_type) {
        case GGML_TYPE_Q8_0:
            ggml_cuda_ar_add_q<block_q8_0, QK8_0, true,
                dequantize_q8_0, quantize_f32_q8_0_block>(stream, dst, src, ne);
            break;
        case GGML_TYPE_Q5_0:
            ggml_cuda_ar_add_q<block_q5_0, QK5_0, false,
                dequantize_q5_0, quantize_f32_q5_0_block>(stream, dst, src, ne);
            break;
        case GGML_TYPE_Q4_0:
            ggml_cuda_ar_add_q<block_q4_0, QK4_0, false,
                dequantize_q4_0, quantize_f32_q4_0_block>(stream, dst, src, ne);
            break;
        default:
            GGML_ASSERT(false);
    }
}

// Byte size of the on-wire buffer for `ne` F32 elements reduced over the
// quantized `wire_type` (ne/QK * block_size).
static inline size_t ggml_cuda_ar_wire_nbytes(int64_t ne, ggml_type wire_type) {
    switch (wire_type) {
        case GGML_TYPE_Q8_0: return (size_t)(ne / QK8_0) * sizeof(block_q8_0);
        case GGML_TYPE_Q5_0: return (size_t)(ne / QK5_0) * sizeof(block_q5_0);
        case GGML_TYPE_Q4_0: return (size_t)(ne / QK4_0) * sizeof(block_q4_0);
        default:             return (size_t) ne * sizeof(float);
    }
}
