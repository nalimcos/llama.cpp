#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// Largest power of two in {64, 128, 256, 512} dividing `ne`, or 0 if none
// does. Used by the AllReduce quantized-wire selector to reject non-rotatable
// tensors before committing to a quantized wire.
int ggml_cuda_ar_fwht_n(int64_t ne);

// Buffer-level Fast Walsh-Hadamard rotation helpers for the internal AllReduce
// quantized-wire path. They operate on a flat contiguous F32 buffer of `ne`
// elements (no ggml_tensor required), rotating in power-of-two chunks.
//
// The transform is its own inverse up to the 1/sqrt(n) normalization, so
// forward and inverse apply the same kernel; both return false when no
// power-of-two chunk length in {64, 128, 256, 512} divides ne (caller then
// falls back to a non-rotated wire).
bool ggml_cuda_ar_fwht_forward(cudaStream_t stream, float * dst, const float * src, int64_t ne);
bool ggml_cuda_ar_fwht_inverse(cudaStream_t stream, float * dst, const float * src, int64_t ne);
