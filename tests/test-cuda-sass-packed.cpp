// Standalone experimental integration test; see docs/build.md for invocation.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-cuda-sass.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) { fprintf(stderr, "%s\n", message); std::exit(1); }
}

static std::vector<float> run(ggml_backend_t backend, int cols, int rows, int tokens, bool linear) {
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    check(ctx != nullptr, "context allocation failed");
    const int rank = 17;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (cols + 31) / 32, linear ? rank : rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor * y;
    std::vector<ggml_tensor *> scales;
    ggml_tensor * u = nullptr;
    if (linear) {
        u = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (rank + 31) / 32, rows);
        for (int n : {rows, rank, rank, cols}) scales.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n));
        y = ggml_cuda_sass_binary_linear(ctx, u, w, x, scales[0], scales[1], scales[2], scales[3]);
    } else {
        y = ggml_cuda_sass_binary_matmul(ctx, w, x);
    }
    check(y != nullptr, "graph API rejected valid shape");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        check(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "backend does not support graph node");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "backend allocation failed");
    for (ggml_tensor * packed : {w, u}) if (packed != nullptr) {
        std::vector<uint32_t> words(ggml_nelements(packed));
        for (size_t i = 0; i < words.size(); ++i) words[i] = 0x9e3779b9U * uint32_t(i + 1);
        ggml_backend_tensor_set(packed, words.data(), 0, ggml_nbytes(packed));
    }
    for (ggml_tensor * scale : scales) {
        std::vector<float> values(ggml_nelements(scale));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 7) - 3) / 8;
        ggml_backend_tensor_set(scale, values.data(), 0, ggml_nbytes(scale));
    }
    std::vector<float> input(cols * tokens), output(rows * tokens);
    // Change input across repeated graph captures/replays to catch stale buffers.
    for (int trial = 0; trial < 5; ++trial) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i + trial) % 31) - 15) / 16;
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    check(cpu && gpu, "backend initialization failed");
    int cases = 0;
    for (int cols : {1, 17, 32, 33, 65, 129}) for (int tokens : {1, 3}) for (bool linear : {false, true}) {
        auto expected = run(cpu, cols, 7, tokens, linear);
        auto actual = run(gpu, cols, 7, tokens, linear);
        check(expected == actual, "packed graph CPU/CUDA mismatch");
        ++cases;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("%d packed graph cases passed (5 executions each, CPU and CUDA)\n", cases);
}
