#include "sampler.h"

#include <vector>
#include <algorithm>

Sampler::Sampler(int32_t vocab_size, float temperature, float top_p, uint32_t seed)
    : vocab_size(vocab_size), temperature(temperature), top_p(top_p), seed(seed) {
    validate();
    engine.seed(seed);
    prob_storage.resize(vocab_size);
    probs = make_tensor(prob_storage.data(), DType::Float32, { vocab_size });
}

void Sampler::validate() const {
    if (vocab_size <= 0) {
        throw std::runtime_error("Invalid sampler config: vocab_size must be positive.");
    }
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::runtime_error("Invalid sampler config: temperature must be positive and finite.");
    }
    if (!std::isfinite(top_p)) {
        throw std::runtime_error("Invalid sampler config: top_p must be finite.");
    }
    if (top_p < 0.0f || top_p > 1.0f) {
        throw std::runtime_error("Invalid sampler config: top_p must be in [0, 1].");
    }
}

float Sampler::random_f32() {
    return dist(engine);
}

uint32_t sample_argmax(const Tensor& logits) {
    assert(logits.data != nullptr);
    assert(logits.ndim == 1);
    assert(logits.shape[0] > 0);

    const int32_t d = logits.shape[0];
    const float* lo_ptr = logits.ptr<float>();

    int32_t max_idx = 0;
    for (int32_t i = 1;i < d;i++) {
        if (lo_ptr[i] - lo_ptr[max_idx] > 0) {
            max_idx = i;
        }
    }
    return (uint32_t)max_idx;
}

uint32_t sample_top_p(Sampler& sampler, const Tensor& logits) {
    assert(logits.data != nullptr);
    assert(logits.ndim == 1);
    assert(logits.shape[0] > 0);
    assert(sampler.top_p >= 0.0f);

    const int32_t d = logits.shape[0];
    const float* lo_ptr = logits.ptr<float>();

    std::vector<std::pair<float, int32_t>> probs;
    probs.resize(d);
    for (int32_t i = 0;i < d;i++) {
        probs[i] = { lo_ptr[i], i };
    }
    std::sort(probs.begin(), probs.end(), std::greater<>());

    float cumsum = 0.0f;
    int32_t last_idx = d - 1;
    for (int32_t i = 0;i < d;i++) {
        cumsum += probs[i].first;
        if (cumsum >= sampler.top_p) {
            last_idx = i;
            break;
        }
    }

    float r = sampler.random_f32() * cumsum;
    cumsum = 0.0f;
    for (int32_t i = 0;i <= last_idx;i++) {
        cumsum += probs[i].first;
        if (cumsum >= r) {
            return (uint32_t)probs[i].second;
        }
    }
    return (uint32_t)probs[last_idx].second;
}


uint32_t sample(Sampler& sampler, const Tensor& logits) {
    copy(logits, sampler.probs);
    if (sampler.temperature == 0.0f) {
        return sample_argmax(sampler.probs);
    }
    div_1d(sampler.probs, sampler.temperature);
    softmax_1d(sampler.probs);
    return sample_top_p(sampler, sampler.probs);
}