#pragma once
#include <cstdint>

struct Config {
    int32_t dim = 2048;
    int32_t ffn_hidden_dim = 0;
    int32_t n_layers = 8;
    int32_t n_heads = 32;
    int32_t n_kv_heads = 4;
    int32_t vocab_size = 32000;
    float norm_eps = 1e-5;
    int32_t max_seq_len = 2048;

    void validate() const;
};
