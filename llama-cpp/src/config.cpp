#include "config.h"

#include <stdexcept>
#include <cmath>

void Config::validate() const {
    if (dim <= 0) {
        throw std::runtime_error("Invalid config: dim must be positive.");
    }
    if (ffn_hidden_dim <= 0) {
        throw std::runtime_error("Invalid config: ffn_hidden_dim must be positive.");
    }
    if (n_layers <= 0) {
        throw std::runtime_error("Invalid config: n_layers must be positive.");
    }
    if (n_heads <= 0) {
        throw std::runtime_error("Invalid config: n_heads must be positive.");
    }
    if (n_kv_heads <= 0) {
        throw std::runtime_error("Invalid config: n_kv_heads must be positive.");
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("Invalid config: vocab_size must be positive.");
    }
    if (max_seq_len <= 0) {
        throw std::runtime_error("Invalid config: max_seq_len must be positive.");
    }
    if (!std::isfinite(norm_eps) || norm_eps <= 0.0f) {
        throw std::runtime_error("Invalid config: norm_eps must be positive and finite.");
    }

    if (dim % n_heads != 0) {
        throw std::runtime_error("Invalid config: dim must be divisible by n_heads.");
    }
    if (n_heads % n_kv_heads != 0) {
        throw std::runtime_error(
            "Invalid config: n_heads must be divisible by n_kv_heads."
        );
    }

    const int32_t head_dim = dim / n_heads;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("Invalid config: head_dim must be even for RoPE.");
    }
}