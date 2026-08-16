#include "model.h"

#include <cmath>

void linear(const Tensor& x, const LinearWeight& w, Tensor& out) {
    switch (w.values.dtype) {
    case DType::Float32:
        matvec(x, w.values, out);break;
    case DType::Int8:
        qmatvec(x, w.values, w.scales, out);break;
    default:
        throw std::runtime_error("Unkown tensor dtype: " + std::to_string(w.values.dtype) + ".");
    }
}

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& out, float eps = 1e-6) {
    assert(x.data != nullptr);
    assert(w.data != nullptr);
    assert(out.data != nullptr);

    assert(x.ndim == 1);
    assert(w.ndim == 1);
    assert(out.ndim == 1);

    const int d = w.shape[0];

    const float* x_ptr = x.ptr<float>();
    const float* w_ptr = w.ptr<float>();
    float* o_ptr = out.ptr<float>();

    float sum = 0;
    for (int i = 0;i < d;i++) {
        sum += x_ptr[i] * x_ptr[i];
    }
    sum = sum / d + eps;
    sum = 1.0f / std::sqrt(sum);

    for (int i = 0;i < d;i++) {
        o_ptr[i] = x_ptr[i] * sum * w_ptr[i];
    }
}

void rotary_embedding(Tensor& x, const int32_t pos, const int32_t head_dim, float base = 10000.0f) {
    assert(x.data != nullptr);
    assert(x.ndim == 1);
    assert(pos >= 0 && head_dim > 0);

    const int32_t dim = x.shape[0];
    assert(head_dim % 2 == 0 && dim % head_dim == 0);
    const int32_t n_heads = dim / head_dim;

    float* x_ptr = x.ptr<float>();

    for (int32_t i = 0;i < head_dim / 2;i++) {
        // theta_i = 1 / base^{2i/d}
        // i : 0, 1, 2, 3, ..., hdim/2
        // 2i: 0, 2, 4, 6, ..., hdim
        const float theta = 1.0f / std::pow(base, 2.0f * i / head_dim);
        const float cos_val = std::cos(theta * pos);
        const float sin_val = std::sin(theta * pos);

        for (int32_t h = 0;h < n_heads;h++) {
            const int32_t offset = h * head_dim;

            const float v0 = x_ptr[offset + i];
            const float v1 = x_ptr[offset + head_dim / 2 + i];
            x_ptr[offset + i] = v0 * cos_val - v1 * sin_val;
            x_ptr[offset + head_dim / 2 + i] = v1 * cos_val + v0 * sin_val;
        }
    }
}

// 单个 token 在某一层中的自注意力计算
// 输入 x: (dim,)，输出 out: (dim,)
// 推理采用增量解码，每次只计算当前位置 pos 的 QKV, K/V 会写入 cache,
// 随后 Q 会与 0..pos 的所有历史 K 做匹配，再用得到的权重汇总历史 V
void Transformer::attention(const Tensor& x, Tensor& out, const int32_t pos, const int32_t layer_idx) {
    assert(config.dim % config.n_heads == 0);
    assert(config.n_heads % config.n_kv_heads == 0);

    // head_size = head_dim 表示每个 attention head 的维度
    const int32_t head_size = config.dim / config.n_heads;
    // kv_dim 是所有 K/V head 拼接后的维度
    // dim    = n_heads    * head_size
    // kv_dim = n_kv_heads * head_size
    const int32_t kv_dim = config.dim * config.n_kv_heads / config.n_heads;
    // kv_mul 表示一个 K/V head 被多少个 Q head 共享
    const int32_t kv_mul = config.n_heads / config.n_kv_heads;

    // 取出当前层的投影权重，以及本层的 K/V cache
    const LayerWeights& layer = weights.layers.at(layer_idx);
    Tensor& q = state.q; // (dim,)

    Tensor& cache_k = state.cache_ks.at(layer_idx); // (max_seq_len, kv_dim)
    Tensor& cache_v = state.cache_vs.at(layer_idx); // (max_seq_len, kv_dim)
    Tensor k = view_row(cache_k, pos); // (kv_dim,)
    Tensor v = view_row(cache_v, pos); // (kv_dim,)

    // 计算当前 token 的 QKV
    linear(x, layer.wq, q); // q = (dim,)    = (n_heads * head_dim)
    linear(x, layer.wk, k); // k = (kv_dim,) = (n_kv_heads * head_dim)
    linear(x, layer.wv, v);

    // 对当前 Q/K 应用 RoPE
    rotary_embedding(q, pos, head_size);
    rotary_embedding(k, pos, head_size);

    // 所有 Q head 的 attention scores 缓冲区
    const Tensor& attn = state.attn; // (n_heads, max_seq_len)
    // 多头上下文
    Tensor& context = state.context; // (dim,)
    fill(context, 0.0f);
    const float scale = 1.0f / sqrt(head_size);

#pragma omp parallel for
    for (int32_t h = 0;h < config.n_heads;h++) {
        // 取出第 h 个 Q head, 以及该 head 在 attn 中的对应的行

        // float* q_ptr = q.ptr<float>() + h * head_size;
        // const int32_t attn_offset = h * config.max_seq_len;
        // float* attn_ptr = attn.ptr<float>() + attn_offset;
        Tensor q_head = view_1d(q, h * head_size, head_size); // (head_size,)
        Tensor attn_head = view_1d(attn, h * config.max_seq_len, config.max_seq_len); // (max_seq_len,)
        float* attn_ptr = attn_head.ptr<float>();

        // 计算当前 Q head 与历史 K 的缩放点积
        // score[h, t] = dot(Q_h(pos), K_{h/kv_mul}(t)) / sqrt(head_size)
        for (int32_t t = 0;t <= pos;t++) {
            Tensor k_t = view_row(cache_k, t); // (kv_dim,)
            // h / kv_mul 把 Q head 映射到共享的 K/V head
            float sorce = dot_1d(q_head, view_1d(k_t, (h / kv_mul) * head_size, head_size));

            attn_ptr[t] = sorce * scale;
        }

        // 对已经存在的历史位置做 softmax
        // softmax_1d(attn, attn_offset, attn_offset + pos + 1);
        softmax_1d(attn_head, 0, pos + 1);

        // 将历史 V 按注意力概率加权求和
        // context_head 指向 context 中第 h 个 head 的输出区域
        // float* c_ptr = context.ptr<float>() + h * head_size;
        Tensor context_head = view_1d(context, h * head_size, head_size); // (head_dim,)

        // context_head = sum_t attn[h, t] * V_{h/kv_mul}(t)
        for (int32_t t = 0;t <= pos;t++) {
            Tensor v_t = view_row(cache_v, t); // (kv_dim,)
            float prob = attn_ptr[t];
            axpy_1d(view_1d(v_t, (h / kv_mul) * head_size, head_size), prob, context_head);
        }
    }

    linear(context, layer.wo, out);
}

void Transformer::layer_forward(const Tensor& x, Tensor& out, const int32_t pos, const int32_t layer_idx) {
    // x: (dim,)
    const int32_t dim = x.shape[0];
    const LayerWeights& layer = weights.layers.at(layer_idx);

    Tensor& norm = state.norm;
    rmsnorm(x, layer.rms_att_weight, norm, config.norm_eps);

    // attention
    Tensor& attn_out = state.attn_out;
    attention(norm, attn_out, pos, layer_idx);

    Tensor& h = state.h;
    // h = x + attention(RMSNorm(x))
    copy(x, h);
    add_1d(attn_out, h);
    // norm_h = RMSNorm(h)
    rmsnorm(h, layer.rms_ffn_weight, norm, config.norm_eps);

    // ffn
    Tensor& h2 = state.h2;
    Tensor& up_h = state.up_h;
    Tensor& down_h = state.down_h;
    // h2 = silu(gate(norm_h)) * up(norm_h)
    // out = down(h2)
    linear(norm, layer.gate, h2);
    silu_1d(h2);
    linear(norm, layer.up, up_h);
    mul_1d(h2, up_h, h2);
    linear(h2, layer.down, down_h);

    // out = h + ffn(RMSNorm(h))
    add_1d(down_h, h);
    copy(h, out);
}

const Tensor& Transformer::forward(const uint32_t idx, const int32_t pos) {
    assert(idx < config.vocab_size);
    assert(0 <= pos && pos < config.max_seq_len);

    Tensor& logits = state.logits;
    Tensor embedding = view_row(weights.embed_tokens, (int32_t)idx);
    copy(embedding, state.x);

    Tensor* cur = &state.x;
    Tensor* nxt = &state.out;

    for (int32_t i = 0;i < config.n_layers;i++) {
        layer_forward(*cur, *nxt, pos, i);
        std::swap(cur, nxt);
    }

    rmsnorm(*cur, weights.rms_norm_weight, *nxt, config.norm_eps);
    linear(*nxt, weights.lm_head, logits);
    return logits;
}