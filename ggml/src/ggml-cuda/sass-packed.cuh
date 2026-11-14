#pragma once
#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_sass_packed_init(ggml_backend_cuda_context * ctx);
void ggml_cuda_sass_packed_free(ggml_backend_cuda_context * ctx);
bool ggml_cuda_sass_packed_supports(const ggml_tensor * op, int device);
bool ggml_cuda_sass_packed_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
