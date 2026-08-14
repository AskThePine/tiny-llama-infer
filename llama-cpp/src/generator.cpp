#include "generator.h"

#include <cmath>

std::vector<uint32_t> Generator::generate(
    const std::vector<uint32_t>& ids, Sampler& sampler, int32_t max_gen_len
) {
    if (max_gen_len < 0) {
        throw std::runtime_error("Invalid config: max_gen_len must be non-negative.");
    }
    if (max_gen_len == 0) {
        return {};
    }
    if (ids.empty()) {
        throw std::runtime_error("Input ids size must be greater than zero.");
    }

    const Config& config = model.config;
    int32_t seq_len = ids.size();
    if (seq_len > config.max_seq_len) {
        throw std::runtime_error("generate sequence must be less than model max_seq_len.");
    }
    int32_t len = std::min(config.max_seq_len - seq_len, max_gen_len);

    Tensor logits{};
    int32_t pos = 0;
    while (pos < seq_len) {
        // 读入 prompt
        logits = model.forward(ids[pos], pos);
        pos++;
    }

    std::vector<uint32_t> gen_ids;
    while (gen_ids.size() < len) {
        const int32_t next_idx = sample(sampler, logits);
        if (next_idx == tokenizer.config.eos_id) {
            break;
        }
        gen_ids.push_back(next_idx);
        if (gen_ids.size() == len) {
            break;
        }
        logits = model.forward(next_idx, pos);
        pos++;
    }

    return gen_ids;
}

std::string Generator::complete(
    const std::string& text, Sampler& sampler, int32_t max_gen_len, bool echo
) {
    auto ids = tokenizer.encode(text);
    const auto gen_ids = generate(ids, sampler, max_gen_len);

    if (echo) {
        ids.insert(ids.end(), gen_ids.begin(), gen_ids.end());
        return tokenizer.decode(ids);
    }
    return tokenizer.decode(gen_ids);
}

