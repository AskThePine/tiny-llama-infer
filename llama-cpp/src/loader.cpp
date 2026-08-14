#include "loader.h"

#include <stdexcept>
#include <cmath>
#include <cstring>

namespace {
    constexpr size_t align_up(size_t value, size_t alignment) noexcept {
        if (alignment == 0) {
            return value;
        }
        size_t remainder = value % alignment;
        if (remainder == 0) {
            return value;
        }
        return value + (alignment - remainder);
    }

    struct Cursor {
        const uint8_t* data;
        size_t size;
        size_t offset;

        Cursor(const void* data, size_t size)
            : data(static_cast<const uint8_t*>(data)), size(size), offset(0) {
        }

        void seek(size_t pos) {
            if (pos > size) {
                throw std::runtime_error("Cursor: out of bounds.");
            }
            offset = pos;
        }

        template<typename T>
        T read() {
            static_assert(std::is_trivial_v<T>, "T must be trivial.");
            if (offset + sizeof(T) > size) {
                throw std::runtime_error("Cursor: out of bounds.");
            }
            T val{};
            std::memcpy(&val, data + offset, sizeof(T));
            offset += sizeof(T);
            return val;
        }

        const uint8_t* read_bytes(size_t n) {
            if (offset + n > size) {
                throw std::runtime_error("Cursor: out of bounds.");
            }
            const uint8_t* p = data + offset;
            offset += n;
            return p;
        }

        std::string read_string(size_t n) {
            const uint8_t* p = read_bytes(n);
            return std::string(reinterpret_cast<const char*>(p), n);
        }

        Tensor take_tensor(const std::initializer_list<int32_t> shape, DType dtype = DType::Float32) {
            if (shape.size() == 0) {
                throw std::runtime_error("Cursor: Tensor shape size must be greater than zero.");
            }
            size_t numel = 1;
            for (int32_t dim : shape) {
                if (dim <= 0) {
                    throw std::runtime_error("Cursor: Dim must be positive.");
                }
                numel *= dim;
            }
            size_t needed = numel * dtype_size(dtype);
            needed = align_up(needed, sizeof(float));

            if (offset + needed > size) {
                throw std::runtime_error("Cursor: Not enough data for tensor.");
            }
            void* tensor_data = const_cast<uint8_t*>(data + offset);
            offset += needed;
            return make_tensor(tensor_data, dtype, shape);
        }

        LinearWeight take_linear_weight(
            const std::initializer_list<int32_t> shape, bool is_quantized = false
        ) {
            if (shape.size() != 2) {
                throw std::runtime_error("LinearWeight shape size must be 2.");
            }
            Tensor values{};
            Tensor scales{};

            if (is_quantized) {
                values = take_tensor(shape, DType::Int8);
                scales = take_tensor({ *shape.begin() }, DType::Float32);
            } else {
                values = take_tensor(shape, DType::Float32);
            }
            return { values, scales };
        }

        bool finished() const {
            return offset == size;
        }
    };

} // namespace

void validate_header(Cursor& cursor) {
    if (cursor.size < HEADER_SIZE) {
        throw std::runtime_error(
            "File size must be greater than header size: " + std::to_string(HEADER_SIZE) + "."
        );
    }
    const uint32_t magic_num = cursor.read<uint32_t>();
    if (magic_num != MAGIC_NUM) {
        throw std::runtime_error("Invalid magic number, file is not valid.");
    }
    const int32_t version = cursor.read<int32_t>();
    if (version != VERSION) {
        throw std::runtime_error("Unsupported version: " + std::to_string(version) + ".");
    }
}

using HeaderFlags = uint8_t;

HeaderFlags load_model_config(Cursor& cursor, Config& config) {
    try {
        validate_header(cursor);
        const uint8_t file_type = cursor.read<uint8_t>();
        if (file_type != MODEL) {
            throw std::runtime_error(
                "Unsupported file type, expect MODEL(0), got: " + std::to_string(file_type) + "."
            );
        }
        const HeaderFlags flags = cursor.read<HeaderFlags>();

        config.dim = cursor.read<int32_t>();
        config.ffn_hidden_dim = cursor.read<int32_t>();
        config.n_layers = cursor.read<int32_t>();
        config.n_heads = cursor.read<int32_t>();
        config.n_kv_heads = cursor.read<int32_t>();
        config.vocab_size = cursor.read<int32_t>();
        config.norm_eps = cursor.read<float>();
        config.max_seq_len = cursor.read<int32_t>();

        cursor.seek(HEADER_SIZE);
        return flags;
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to read model config header from file: " + std::string(e.what()));
    }
}

void load_weights(Cursor& cursor, Transformer& transformer, HeaderFlags flags) {
    const Config& config = transformer.config;
    const int32_t dim = config.dim;
    const int32_t kv_dim = config.n_kv_heads * config.dim / config.n_heads;
    const int32_t h_dim = config.ffn_hidden_dim;
    const bool is_quantized = flags & 0x01;

    TransformerWeights& weights = transformer.weights;
    // token embedding table
    weights.embed_tokens = cursor.take_tensor({ config.vocab_size, dim });
    // layer weights
    weights.layers.resize(config.n_layers);
    for (int32_t i = 0; i < config.n_layers;i++) {
        LayerWeights& layer = weights.layers.at(i);
        // attention weights
        layer.wq = cursor.take_linear_weight({ dim, dim }, is_quantized);
        layer.wk = cursor.take_linear_weight({ kv_dim, dim }, is_quantized);
        layer.wv = cursor.take_linear_weight({ kv_dim, dim }, is_quantized);
        layer.wo = cursor.take_linear_weight({ dim, dim }, is_quantized);
        // rmsnorm weigths
        layer.rms_att_weight = cursor.take_tensor({ dim });
        layer.rms_ffn_weight = cursor.take_tensor({ dim });
        // ffn weigths
        layer.gate = cursor.take_linear_weight({ h_dim, dim }, is_quantized);
        layer.up = cursor.take_linear_weight({ h_dim, dim }, is_quantized);
        layer.down = cursor.take_linear_weight({ dim, h_dim }, is_quantized);
    }
    // header weigths
    weights.lm_head = cursor.take_linear_weight({ config.vocab_size, dim }, is_quantized);
    // rmsnorm weigths
    weights.rms_norm_weight = cursor.take_tensor({ dim });
}

void load_runstate(Transformer& transformer) {
    const Config& config = transformer.config;
    const int32_t dim = config.dim;
    const int32_t kv_dim = config.n_kv_heads * config.dim / config.n_heads;
    const int32_t h_dim = config.ffn_hidden_dim;
    const int32_t vocab_size = config.vocab_size;
    const int32_t max_seq_len = config.max_seq_len;

    const size_t size = (size_t)dim * 8 + h_dim * 2
        + config.n_heads * max_seq_len + vocab_size
        + 2 * config.n_layers * max_seq_len * kv_dim;
    Storage& storage = transformer.storage;
    try {
        storage.state.resize(size);
    }
    catch (const std::bad_alloc& e) {
        throw std::runtime_error(
            "Insufficient memory to load runstate (size: " + std::to_string(size) + " floats)."
        );
    }

    Cursor cursor(storage.state.data(), storage.state.size() * sizeof(float));
    RunState& state = transformer.state;
    state.norm = cursor.take_tensor({ dim });
    state.h = cursor.take_tensor({ dim });
    state.h2 = cursor.take_tensor({ h_dim });
    state.up_h = cursor.take_tensor({ h_dim });
    state.down_h = cursor.take_tensor({ dim });
    state.q = cursor.take_tensor({ dim });
    state.context = cursor.take_tensor({ dim });
    state.attn = cursor.take_tensor({ config.n_heads * max_seq_len });
    state.attn_out = cursor.take_tensor({ dim });
    state.x = cursor.take_tensor({ dim });
    state.out = cursor.take_tensor({ dim });
    state.logits = cursor.take_tensor({ vocab_size });

    state.cache_ks.resize(config.n_layers);
    state.cache_vs.resize(config.n_layers);
    for (int32_t i = 0;i < config.n_layers;i++) {
        state.cache_ks.at(i) = cursor.take_tensor({ max_seq_len,kv_dim });
        state.cache_vs.at(i) = cursor.take_tensor({ max_seq_len,kv_dim });
    }

    if (!cursor.finished()) {
        throw std::runtime_error("RunState storage not fully bound.");
    }
}

void load_transformer(const std::string& file_path, Transformer& transformer) {
    transformer.storage.model = MappedFile(file_path);
    Cursor cursor(transformer.storage.model.data(), transformer.storage.model.size());
    const auto flags = load_model_config(cursor, transformer.config);
    transformer.config.validate();
    load_weights(cursor, transformer, flags);
    if (!cursor.finished()) {
        throw std::runtime_error("Transformer not fully loaded.");
    }
    load_runstate(transformer);
}

void load_tokenizer_config(Cursor& cursor, TokenizerConfig& config) {
    try {
        validate_header(cursor);
        const uint8_t file_type = cursor.read<uint8_t>();
        if (file_type != TOKENIZER) {
            throw std::runtime_error(
                "Unsupported file type, expect TOKENIZER(1), got: " + std::to_string(file_type) + "."
            );
        }

        config.vocab_size = cursor.read<int32_t>();
        config.merge_count = cursor.read<int32_t>();
        config.model_max_length = cursor.read<int32_t>();
        config.bos_id = cursor.read<uint32_t>();
        config.eos_id = cursor.read<uint32_t>();
        config.pad_id = cursor.read<uint32_t>();
        config.unk_id = cursor.read<uint32_t>();

        cursor.seek(HEADER_SIZE);
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to read tokenizer config header from file: " + std::string(e.what()));
    }
}

void load_vocab(Cursor& cursor, Tokenizer& tokenizer) {
    const int32_t size = tokenizer.config.vocab_size;
    Vocab& token_to_id = tokenizer.token_to_id;
    std::vector<std::string>& id_to_token = tokenizer.id_to_token;
    token_to_id.clear();
    id_to_token.assign(size, {});

    for (int32_t i = 0;i < size;i++) {
        const uint32_t token_size = cursor.read<uint32_t>();
        const uint32_t id = cursor.read<uint32_t>();
        if (id >= (uint32_t)size) {
            throw std::runtime_error(
                "Token id: " + std::to_string(id) + "is greater than vocab_size: " + std::to_string(size) + "."
            );
        }

        std::string token = cursor.read_string((size_t)token_size);
        if (token_to_id.count(token) != 0) {
            throw std::runtime_error("Tokenizer vocab has the same token: " + token + ".");
        }
        if (!id_to_token[id].empty()) {
            throw std::runtime_error("Tokenizer vocab has the same id: " + std::to_string(id) + ".");
        }
        token_to_id[token] = id;
        id_to_token[id] = token;
    }

    if (token_to_id.size() != size) {
        throw std::runtime_error("Tokenizer vocab size is not equal to tokenizer.config.vocab_size.");
    }
}

void load_pair_rank(Cursor& cursor, Tokenizer& tokenizer) {
    const int32_t count = tokenizer.config.merge_count;
    Rank& pair_rank = tokenizer.pair_rank;
    pair_rank.clear();

    for (int32_t i = 0;i < count;i++) {
        const uint32_t rank = cursor.read<uint32_t>();
        const uint32_t left_size = cursor.read<uint32_t>();
        const uint32_t right_size = cursor.read<uint32_t>();
        if (rank >= count) {
            throw std::runtime_error(
                "Rank: " + std::to_string(rank) + "is greater than pair_rank size: " + std::to_string(count) + "."
            );
        }

        std::string left = cursor.read_string((size_t)left_size);
        std::string right = cursor.read_string((size_t)right_size);

        if (tokenizer.token_to_id.count(left) == 0) {
            throw std::runtime_error("Left token: " + left + " is not in tokenzier voacb.");
        }
        if (tokenizer.token_to_id.count(right) == 0) {
            throw std::runtime_error("Right token: " + right + " is not in tokenzier voacb.");
        }
        const Pair pair = { left, right };
        if (pair_rank.count(pair) != 0) {
            throw std::runtime_error(
                "Tokenizer pair_rank has the same pair: (" + left + ", " + right + ")."
            );
        }
        pair_rank[pair] = rank;
    }
}

void load_tokenizer(const std::string& file_path, Tokenizer& tokenizer) {
    MappedFile file(file_path);
    Cursor cursor(file.data(), file.size());
    load_tokenizer_config(cursor, tokenizer.config);
    tokenizer.config.vaildate();
    load_vocab(cursor, tokenizer);
    tokenizer.init_byte_fallback();
    load_pair_rank(cursor, tokenizer);
    if (!cursor.finished()) {
        throw std::runtime_error("Tokenizer not fully loaded.");
    }
}