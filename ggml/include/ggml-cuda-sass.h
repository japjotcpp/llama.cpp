#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental graph API. Packed I32 rows use bit 0 first: 1 is -1, 0 is +1.
// packed: [ceil(cols/32), rows], x: [cols, tokens], result: FP32 [rows, tokens].
// Inputs must be contiguous. Invalid shapes/types return NULL. CPU fallback is provided.
GGML_BACKEND_API struct ggml_tensor * ggml_cuda_sass_binary_matmul(
        struct ggml_context * ctx, struct ggml_tensor * packed, struct ggml_tensor * x);

// One FP32 LittleBit-compatible branch: u1 * U * ((v1*u2) * (V * (v2*x))).
// U/V remain packed; all scales are contiguous FP32 vectors. No checkpoint loader is implied.
GGML_BACKEND_API struct ggml_tensor * ggml_cuda_sass_binary_linear(
        struct ggml_context * ctx, struct ggml_tensor * u, struct ggml_tensor * v,
        struct ggml_tensor * x, struct ggml_tensor * u1, struct ggml_tensor * u2,
        struct ggml_tensor * v1, struct ggml_tensor * v2);

#ifdef __cplusplus
}
#endif
