#pragma once
#include "tensor.h"

#include <random>

struct Sampler {
    int32_t vocab_size;

    float temperature;
    float top_p;

    std::vector<float> prob_storage;
    Tensor probs;

    uint32_t seed;
    std::mt19937 engine;
    std::uniform_real_distribution<float> dist{ 0.0f, 1.0f };

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(int32_t vocab_size, float temperature = 0.0f, float top_p = 0.9f, uint32_t seed = 42);
    void validate() const;
    float random_f32();
};

uint32_t sample(Sampler& sampler, const Tensor& logits);