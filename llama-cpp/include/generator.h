#pragma once
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

struct Generator {
    Transformer& model;
    Tokenizer& tokenizer;

    Generator(Transformer& model, Tokenizer& tokenizer) : model(model), tokenizer(tokenizer) {}

    std::vector<uint32_t> generate(
        const std::vector<uint32_t>& ids, Sampler& sampler, int32_t max_gen_len = 64
    );

    std::string complete(
        const std::string& text, Sampler& sampler, int32_t max_gen_len = 64, bool echo = false
    );
};