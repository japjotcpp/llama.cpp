#include "sass-packed.cuh"
#include "ggml-cuda-sass.h"
#include "graph.cuh"

#include <cstring>
#include <cinttypes>
#include <limits>
#include <mutex>

static bool packed_shapes(const ggml_tensor * packed, const ggml_tensor * x) {
    if (packed == nullptr || x == nullptr || packed->type != GGML_TYPE_I32 || x->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[0] < 1 || x->ne[0] > INT32_MAX - 31 || x->ne[1] < 1 || x->ne[1] > INT32_MAX ||
        packed->ne[1] < 1 || packed->ne[1] > INT32_MAX / 32 ||
        packed->ne[0] != (x->ne[0] + 31) / 32 ||
        packed->ne[2] != 1 || packed->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    const uint64_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
    if (uint64_t(x->ne[0]) > max_elements / uint64_t(x->ne[1]) ||
        uint64_t(packed->ne[1]) > max_elements / uint64_t(x->ne[1]) ||
        uint64_t(packed->ne[0]) > max_elements / uint64_t(packed->ne[1])) {
        return false;
    }
    return ggml_is_contiguous(packed) && ggml_is_contiguous(x);
}

static void packed_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * packed = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    GGML_ASSERT(packed_shapes(packed, x));
    const auto * words = static_cast<const uint32_t *>(packed->data);
    const auto * input = static_cast<const float *>(x->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t rows = packed->ne[1];
    const int64_t cols = x->ne[0];
    const int64_t stride = packed->ne[0];
    for (int64_t index = ith; index < rows * x->ne[1]; index += nth) {
        const int64_t row = index % rows;
        const int64_t token = index / rows;
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            const bool negative = (words[row * stride + col / 32] >> (col % 32)) & 1U;
            const float value = input[token * cols + col];
            sum += negative ? -value : value;
        }
        output[index] = sum;
    }
}

ggml_tensor * ggml_cuda_sass_binary_matmul(ggml_context * ctx, ggml_tensor * packed, ggml_tensor * x) {
    if (!packed_shapes(packed, x)) {
        return nullptr;
    }
    ggml_tensor * args[] = { packed, x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, packed->ne[1], x->ne[1], 1, 1,
                          args, 2, packed_cpu, GGML_N_TASKS_MAX, nullptr);
}

static bool packed_scale(const ggml_tensor * scale, int64_t length) {
    return scale != nullptr && scale->type == GGML_TYPE_F32 && scale->ne[0] == length &&
           scale->ne[1] == 1 && scale->ne[2] == 1 && scale->ne[3] == 1 && ggml_is_contiguous(scale);
}

ggml_tensor * ggml_cuda_sass_binary_linear(ggml_context * ctx, ggml_tensor * u, ggml_tensor * v,
        ggml_tensor * x, ggml_tensor * u1, ggml_tensor * u2, ggml_tensor * v1, ggml_tensor * v2) {
    if (!packed_shapes(v, x) || u == nullptr || u->type != GGML_TYPE_I32 ||
        u->ne[0] != (v->ne[1] + 31) / 32 || u->ne[1] < 1 || u->ne[1] > INT32_MAX / 32 ||
        u->ne[2] != 1 || u->ne[3] != 1 || !ggml_is_contiguous(u) ||
        !packed_scale(u1, u->ne[1]) || !packed_scale(u2, v->ne[1]) ||
        !packed_scale(v1, v->ne[1]) || !packed_scale(v2, x->ne[0])) {
        return nullptr;
    }
    ggml_tensor * scaled_x = ggml_mul(ctx, x, v2);
    ggml_tensor * hidden = ggml_cuda_sass_binary_matmul(ctx, v, scaled_x);
    ggml_tensor * middle_scale = ggml_mul(ctx, v1, u2);
    hidden = ggml_mul(ctx, hidden, middle_scale);
    ggml_tensor * result = ggml_cuda_sass_binary_matmul(ctx, u, hidden);
    return result == nullptr ? nullptr : ggml_mul(ctx, result, u1);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct packed_module {
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    int device;
    uint64_t dispatches;
    bool trace;
};

static std::mutex packed_mutex;
static std::unordered_map<ggml_backend_cuda_context *, packed_module> packed_modules;

static bool packed_node(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM || op->type != GGML_TYPE_F32) {
        return false;
    }
    ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    return params.fun == packed_cpu && params.userdata == nullptr &&
           packed_shapes(op->src[0], op->src[1]) && ggml_is_contiguous(op) &&
           op->ne[0] == op->src[0]->ne[1] && op->ne[1] == op->src[1]->ne[1] &&
           op->ne[2] == 1 && op->ne[3] == 1;
}

void ggml_cuda_sass_packed_init(ggml_backend_cuda_context * ctx) {
    const char * path = std::getenv("GGML_CUDA_SASS_BINARY_CUBIN");
    if (path == nullptr || path[0] == '\0' || ggml_cuda_info().devices[ctx->device].cc != GGML_CUDA_CC_TURING) {
        return;
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaFree(nullptr));
    packed_module state{};
    state.device = ctx->device;
    const char * trace = std::getenv("GGML_CUDA_SASS_TRACE");
    state.trace = trace != nullptr && std::strcmp(trace, "0") != 0;
    CU_CHECK(cuCtxGetCurrent(&state.context));
    CUresult status = cuModuleLoad(&state.module, path);
    if (status == CUDA_SUCCESS) {
        status = cuModuleGetFunction(&state.kernel, state.module, "binary_dot");
    }
    int binary_version = 0;
    if (status == CUDA_SUCCESS) {
        status = cuFuncGetAttribute(&binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, state.kernel);
    }
    if (status != CUDA_SUCCESS || binary_version != 75) {
        GGML_LOG_WARN("%s: cannot load SM75 binary_dot from '%s'; packed graph nodes use CPU fallback\n", __func__, path);
        if (state.module != nullptr) {
            CU_CHECK(cuModuleUnload(state.module));
        }
        return;
    }
    std::lock_guard<std::mutex> lock(packed_mutex);
    packed_modules.emplace(ctx, state);
    GGML_LOG_INFO("%s: experimental packed binary graph kernel loaded on device %d\n", __func__, ctx->device);
}

bool ggml_cuda_sass_packed_supports(const ggml_tensor * op, int device) {
    if (!packed_node(op)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(packed_mutex);
    for (const auto & entry : packed_modules) {
        if (entry.second.device == device) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_sass_packed_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!packed_node(dst)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(packed_mutex);
    auto found = packed_modules.find(&ctx);
    if (found == packed_modules.end()) {
        return false;
    }
    packed_module & state = found->second;
    CUcontext current = nullptr;
    CU_CHECK(cuCtxGetCurrent(&current));
    GGML_ASSERT(current == state.context);
    const ggml_tensor * packed = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const void * weights = packed->data;
    int32_t rows = int32_t(packed->ne[1]);
    int32_t cols = int32_t(x->ne[0]);
    for (int64_t token = 0; token < x->ne[1]; ++token) {
        const void * input = static_cast<const char *>(x->data) + token * x->nb[1];
        void * output = static_cast<char *>(dst->data) + token * dst->nb[1];
        void * params[] = { &weights, &input, &output, &rows, &cols };
        CU_CHECK(cuLaunchKernel(state.kernel, (uint32_t(rows) + 3) / 4, 1, 1, 128, 1, 1, 0, ctx.stream(), params, nullptr));
        ++state.dispatches;
        if (state.trace && state.dispatches == 1) {
            GGML_LOG_INFO("%s: SASS binary_dot host dispatch rows=%d cols=%d tokens=%" PRId64 "\n", __func__, rows, cols, x->ne[1]);
        }
    }
    return true;
}

void ggml_cuda_sass_packed_free(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(packed_mutex);
    auto found = packed_modules.find(ctx);
    if (found == packed_modules.end()) {
        return;
    }
    packed_module state = found->second;
    CU_CHECK(cuCtxPushCurrent(state.context));
    CU_CHECK(cuCtxSynchronize());
#ifdef USE_CUDA_GRAPH
    ctx->cuda_graphs.clear();
#endif
    CU_CHECK(cuModuleUnload(state.module));
    CUcontext previous = nullptr;
    CU_CHECK(cuCtxPopCurrent(&previous));
    if (state.trace) {
        GGML_LOG_INFO("%s: SASS binary_dot host_dispatches=%" PRIu64 " (includes captures; excludes replays)\n", __func__, state.dispatches);
    }
    packed_modules.erase(found);
}
#endif
