#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>

using TokenId = uint32_t;

struct TokenizerConfig {
    int32_t vocab_size = 32000;
    int32_t merge_count = 61249;
    int32_t model_max_length = 2048;
    // special token ids
    TokenId bos_id = 1;
    TokenId eos_id = 2;
    TokenId pad_id = 2;
    TokenId unk_id = 0;

    void validate() const;
};

struct TokenPair {
    TokenId left;
    TokenId right;
    bool operator==(const TokenPair& other) const {
        return left == other.left && right == other.right;
    }
};

struct TokenPairHash {
    std::size_t operator()(const TokenPair& pair) const noexcept {
        std::size_t h1 = std::hash<TokenId>{}(pair.left);
        std::size_t h2 = std::hash<TokenId>{}(pair.right);
        return h1 ^ (h2 << 1);
    }
};

struct MergeRule {
    TokenId merged_id;
    uint32_t rank;
};

using Vocab = std::unordered_map<std::string, TokenId>;
using MergeTable = std::unordered_map<TokenPair, MergeRule, TokenPairHash>;
using ByteTokens = std::array<TokenId, 256>;
using TokenIds = std::vector<TokenId>;

struct Tokenizer {
    TokenizerConfig config;

    Vocab token_to_id;
    std::vector<std::string> id_to_token;
    MergeTable merge_rules;

    ByteTokens byte_tokens;

    void init_byte_fallback();

    bool is_special_token(const TokenId id) const;

    TokenIds encode(const std::string& text, bool add_bos = true, bool add_eos = false) const;
    std::string decode(const TokenIds& ids, bool skip_special_tokens = true) const;
};