#pragma once
#include "config.h"
#include "tensor.h"
#include "mapped_file.h"

#include <vector>

struct LinearWeight {
    Tensor values; // Float32 or Int8
    Tensor scales; // Float32 (only for quantization)
};

struct LayerWeights {
    // attention weights
    LinearWeight wq; // (n_heads * head_dim, dim)
    LinearWeight wk; // (n_kv_heads * head_dim, dim)
    LinearWeight wv; // (n_kv_heads * head_dim, dim)
    LinearWeight wo; // (n_heads * head_dim, dim)
    // rmsnorm weigths
    Tensor rms_att_weight; // (dim,)
    Tensor rms_ffn_weight; // (dim,)
    // ffn weigths
    LinearWeight gate; // (hidden_dim, dim)
    LinearWeight up; // (hidden_dim, dim)
    LinearWeight down; // (dim, hidden_dim)
};

struct TransformerWeights {
    // token embedding table
    Tensor embed_tokens; // (vocab_size, dim)
    // layer weights
    std::vector<LayerWeights> layers; // (layer,)
    // header weigths
    LinearWeight lm_head; // (vocab_size, dim)
    // rmsnorm weigths
    Tensor rms_norm_weight; // (dim,)
};

struct RunState {
    Tensor norm;   // (dim,)
    Tensor h;      // (dim,)
    Tensor h2;     // (hidden_dim,)
    Tensor up_h;   // (hidden_dim,)
    Tensor down_h; // (dim,)

    Tensor q; // (dim,)

    Tensor context;  // (dim,)
    Tensor attn;     // (n_heads * max_seq_len)
    Tensor attn_out; // (dim,)

    Tensor x; // (dim,)
    Tensor out; // (dim,)
    Tensor logits; // (vocab_size,)

    std::vector<Tensor> cache_ks; // (layer, max_seq_len, kv_dim)
    std::vector<Tensor> cache_vs; // (layer, max_seq_len, kv_dim)
};

struct Storage {
    MappedFile model;
    std::vector<float> state;
};

struct Transformer {
    Config config;
    TransformerWeights weights;
    RunState state;
    Storage storage;

    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    void attention(const Tensor& x, Tensor& out, const int32_t pos, const int32_t layer_idx);
    void layer_forward(const Tensor& x, Tensor& out, const int32_t pos, const int32_t layer_idx);
    const Tensor& forward(const uint32_t idx, const int32_t pos);
};
