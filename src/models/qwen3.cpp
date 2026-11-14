#include "models.h"

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    uint32_t littlebit_version = 0;
    ml.get_key("qwen3.littlebit.version", littlebit_version, false);
    if (littlebit_version > 1) {
        throw std::runtime_error("unsupported Qwen3 LittleBit storage version");
    }
#ifndef LLAMA_SASS_SM75
    if (littlebit_version != 0) {
        throw std::runtime_error("LittleBit Qwen3 requires the experimental CUDA SASS build");
    }
#endif
    auto projection = [&](llm_tensor tensor, int layer, int64_t input, int64_t output) -> ggml_tensor * {
        if (littlebit_version == 0) {
            return create_tensor(tn(tensor, "weight", layer), {input, output}, 0);
        }
        llama_littlebit_projection p;
        auto branch = [&](const char * suffix, llama_littlebit_branch & b, bool required) {
            const std::string prefix = std::string("littlebit.") + suffix;
            const std::string vname = prefix + "v";
            const ggml_tensor * meta = ml.get_tensor_meta(tn(tensor, vname.c_str(), layer).str().c_str());
            if (meta == nullptr && !required) {
                return;
            }
            if (meta == nullptr || meta->type != GGML_TYPE_I32 || meta->ne[0] != (input + 31) / 32 ||
                meta->ne[1] < 1 || meta->ne[1] > INT32_MAX / 32 || meta->ne[2] != 1 || meta->ne[3] != 1) {
                throw std::runtime_error("invalid LittleBit packed V shape/type");
            }
            const int64_t rank = meta->ne[1];
            auto load = [&](const char * name, std::initializer_list<int64_t> shape, ggml_type type) {
                const std::string field = prefix + name;
                ggml_tensor * result = create_tensor(tn(tensor, field.c_str(), layer), shape, 0);
                if (result->type != type) {
                    throw std::runtime_error("invalid LittleBit tensor storage type");
                }
                return result;
            };
            b.v = load("v", {(input + 31) / 32, rank}, GGML_TYPE_I32);
            b.u = load("u", {(rank + 31) / 32, output}, GGML_TYPE_I32);
            b.v2 = load("v2", {input}, GGML_TYPE_F32);
            b.v1 = load("v1", {rank}, GGML_TYPE_F32);
            b.u2 = load("u2", {rank}, GGML_TYPE_F32);
            b.u1 = load("u1", {output}, GGML_TYPE_F32);
        };
        branch("", p.main, true);
        branch("residual.", p.residual, false);
        p.bias = create_tensor(tn(tensor, "littlebit.bias", layer), {output}, TENSOR_NOT_REQUIRED);
        if (p.bias != nullptr && p.bias->type != GGML_TYPE_F32) {
            throw std::runtime_error("LittleBit bias must be FP32");
        }
        littlebit_projections.emplace(p.main.v, p);
        return p.main.v;
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    // output embedding projection
    dense_2_out_layers   = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.n_embd_out()}, TENSOR_NOT_REQUIRED);
    dense_2_out_layers_b = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "bias"),   {hparams.n_embd_out()        }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (littlebit_version == 0) {
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        } else {
            layer.wq = projection(LLM_TENSOR_ATTN_Q, i, n_embd, n_embd_head_k * n_head);
            layer.wk = projection(LLM_TENSOR_ATTN_K, i, n_embd, n_embd_gqa);
            layer.wv = projection(LLM_TENSOR_ATTN_V, i, n_embd, n_embd_gqa);
        }
        layer.wo = projection(LLM_TENSOR_ATTN_OUT, i, n_embd_head_k * n_head, n_embd);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = projection(LLM_TENSOR_FFN_GATE, i, n_embd, n_ff);
        layer.ffn_down = projection(LLM_TENSOR_FFN_DOWN, i, n_ff, n_embd);
        layer.ffn_up   = projection(LLM_TENSOR_FFN_UP, i, n_embd, n_ff);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_qwen3::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_backbone();

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_qwen3::graph::build_backbone() {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = build_rope(Qcur, inp_pos, il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    return cur;
}
