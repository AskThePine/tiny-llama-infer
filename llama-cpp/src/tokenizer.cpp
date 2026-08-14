#include "tokenizer.h"

#include <stdexcept>

#include <unicode/utf8.h>
#include <unicode/uchar.h>

void TokenizerConfig::validate() const {
    if (vocab_size <= 0) {
        throw std::runtime_error("Invalid tokenizer config: vocab_size must be positive.");
    }
    if (merge_count <= 0) {
        throw std::runtime_error("Invalid tokenizer config: merge_count must be positive.");
    }
    if (model_max_length <= 0) {
        throw std::runtime_error("Invalid tokenizer config: model_max_length must be positive.");
    }
    if (bos_id >= vocab_size) {
        throw std::runtime_error("Invalid tokenizer config: bos_id must not be greater than vocab_size.");
    }
    if (eos_id >= vocab_size) {
        throw std::runtime_error("Invalid tokenizer config: eos_id must not be greater than vocab_size.");
    }
    if (pad_id >= vocab_size) {
        throw std::runtime_error("Invalid tokenizer config: pad_id must not be greater than vocab_size.");
    }
    if (unk_id >= vocab_size) {
        throw std::runtime_error("Invalid tokenizer config: unk_id must not be greater than vocab_size.");
    }
}


constexpr uint32_t kSpaceCodepoint = 0x20;
constexpr char kSpaceChar = ' ';
constexpr char kReplacementStr[] = "\xEF\xBF\xBD";

bool is_whitespace(const uint32_t cp) {
    return cp == kSpaceCodepoint;
}

bool is_unicode_whitespace(const uint32_t cp) {
    return u_isUWhiteSpace(static_cast<UChar32>(cp));
}

std::vector<uint32_t> decode_utf8(const std::string& text) {
    std::vector<uint32_t> res;
    const int32_t length = text.size();
    const uint8_t* s = reinterpret_cast<const uint8_t*>(text.data());

    int32_t i = 0;
    while (i < length) {
        UChar32 cp{};
        // 读取一个完整的 UTF-8 code point
        U8_NEXT(s, i, length, cp);
        if (cp < 0) {
            throw std::runtime_error("Invalid UTF-8 sequence.");
        }
        res.push_back(static_cast<uint32_t>(cp));
    }
    return res;
}

std::string encode_utf8(const uint32_t cp) {
    std::string res;
    char buf[4];
    int32_t i = 0;
    bool isError = false;
    U8_APPEND(reinterpret_cast<uint8_t*>(buf), i, 4, static_cast<UChar32>(cp), isError);
    res.append(buf, i);
    return res;
}

using Symbols = std::vector<std::string>;

Symbols pre_tokenizer(const std::string& text) {
    const auto codepoints = decode_utf8(text);

    Symbols symbols;
    if (!codepoints.empty() && !is_whitespace(codepoints.front())) {
        symbols.emplace_back(1, kSpaceChar);
    }
    for (const auto cp : codepoints) {
        if (is_whitespace(cp)) {
            symbols.emplace_back(1, kSpaceChar);
        } else {
            symbols.push_back(encode_utf8(cp));
        }
    }

    return symbols;
}

TokenIds byte_fallback(
    const Symbols& symbols, const Vocab& token_to_id, const ByteTokens& byte_tokens
) {
    TokenIds res;
    for (const auto& symbol : symbols) {
        auto it = token_to_id.find(symbol);
        if (it != token_to_id.end()) {
            res.push_back(it->second);
            continue;
        }
        for (uint8_t c : symbol) {
            res.push_back(byte_tokens[c]);
        }
    }
    return res;
}

TokenIds merge_pair(const TokenIds& tokens, const TokenPair& best, const TokenId merged_id) {
    const size_t size = tokens.size();
    if (size <= 1) {
        return tokens;
    }

    TokenIds res;
    for (size_t i = 0;i < size;) {
        if (i < size - 1) {
            const TokenPair pair = { tokens[i], tokens[i + 1] };
            if (pair == best) {
                res.push_back(merged_id);
                i += 2;
                continue;
            }
        }
        res.push_back(tokens[i]);
        i++;
    }
    return res;
}

TokenIds apply_bpe(const TokenIds& tokens, const MergeTable& merges) {
    TokenIds cur = tokens;
    while (true) {
        TokenPair best{};
        uint32_t min_rank = UINT32_MAX;
        TokenId merged_id = 0;
        for (size_t i = 1;i < cur.size();i++) {
            const TokenPair pair{ cur[i - 1], cur[i] };
            auto it = merges.find(pair);
            if (it != merges.end() && it->second.rank < min_rank) {
                best = pair;
                min_rank = it->second.rank;
                merged_id = it->second.merged_id;
            }
        }
        if (min_rank == UINT32_MAX) {
            break;
        }

        cur = merge_pair(cur, best, merged_id);
        if (cur.size() <= 1) {
            break;
        }
    }
    return cur;
}

constexpr char hex[] = "0123456789ABCDEF";

void Tokenizer::init_byte_fallback() {
    std::string s = "<0x00>";
    for (size_t i = 0;i < 16;i++) {
        for (size_t j = 0;j < 16;j++) {
            s[3] = hex[i];
            s[4] = hex[j];
            auto it = token_to_id.find(s);
            if (it == token_to_id.end()) {
                throw std::runtime_error("Missing byte fallback token: " + s + ".");
            }
            byte_tokens[i * 16 + j] = it->second;
        }
    }
}

bool Tokenizer::is_special_token(const TokenId id) const {
    return id == config.bos_id || id == config.eos_id || id == config.pad_id;
}

TokenIds Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<uint32_t> ids;
    if (add_bos) {
        ids.push_back(config.bos_id);
    }

    auto symbols = pre_tokenizer(text);
    auto initial_ids = byte_fallback(symbols, token_to_id, byte_tokens);
    auto merged_ids = apply_bpe(initial_ids, merge_rules);
    for (const auto& id : merged_ids) {
        ids.push_back(id);
    }

    if (add_eos) {
        ids.push_back(config.eos_id);
    }
    return ids;
}

bool parse_byte_token(const std::string& s, uint8_t& byte) {
    if (s.size() != 6) {
        return false;
    }
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') {
        return false;
    }
    uint8_t val = 0, digit = 0;
    for (size_t i = 3;i < 5;i++) {
        char c = s[i];
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else {
            return false;
        }
        val = val * 16 + digit;
    }
    byte = val;
    return true;
}

void flush_byte_buffer(std::string& res, std::vector<uint8_t>& buffer) {
    if (buffer.empty()) {
        return;
    }

    bool valid_utf8 = true;
    const int32_t length = buffer.size();
    int32_t i = 0;
    while (i < length) {
        UChar32 cp{};
        U8_NEXT(buffer.data(), i, length, cp);
        if (cp < 0) {
            valid_utf8 = false;
            break;
        }
    }

    if (valid_utf8) {
        res.append(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    } else {
        for (size_t i = 0;i < buffer.size();i++) {
            res.append(kReplacementStr);
        }
    }
    buffer.clear();
}

std::string Tokenizer::decode(const TokenIds& ids, bool skip_special_tokens) const {
    std::string res;
    std::vector<uint8_t> buffer;
    uint8_t byte = 0;
    for (const auto& id : ids) {
        if (id >= id_to_token.size()) {
            throw std::runtime_error("Invalid token id: " + std::to_string(id) + ".");
        }
        // TinyLlama 不会跳过 <unk>
        if (skip_special_tokens && is_special_token(id)) {
            continue;
        }
        const std::string& token = id_to_token[id];
        if (parse_byte_token(token, byte)) {
            buffer.push_back(byte);
        } else {
            flush_byte_buffer(res, buffer);
            res.append(token);
        }
    }
    flush_byte_buffer(res, buffer);

    if (res.size() > 0 && res.front() == kSpaceChar) {
        res.erase(0, 1);
    }
    return res;
}
