#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <map>
#include <array>
#include <string>

struct TokenizerConfig {
    int32_t vocab_size = 32000;
    int32_t merge_count = 61249;
    int32_t model_max_length = 2048;
    // special token ids
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    uint32_t pad_id = 2;
    uint32_t unk_id = 0;

    void vaildate() const;
};

using Vocab = std::unordered_map<std::string, uint32_t>;
using Pair = std::array<std::string, 2UL>;
using Rank = std::map<Pair, uint32_t>;
using ByteTokens = std::array<std::string, 256>;

struct Tokenizer {
    TokenizerConfig config;

    Vocab token_to_id;
    std::vector<std::string> id_to_token;
    Rank pair_rank;

    ByteTokens byte_tokens;

    void init_byte_fallback();

    bool is_special_token(const uint32_t id) const;

    std::vector<uint32_t> encode(
        const std::string& text, bool add_bos = true, bool add_eos = false
    );
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens = true) const;
};