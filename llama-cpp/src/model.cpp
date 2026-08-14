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

void Transformer::attention(const Tensor& x, Tensor& out, const int32_t pos, const int32_t layer_idx) {
    assert(config.dim % config.n_heads == 0);
    assert(config.n_heads % config.n_kv_heads == 0);

    const int32_t head_size = config.dim / config.n_heads;
    const int32_t kv_dim = config.dim * config.n_kv_heads / config.n_heads;
    const int32_t kv_mul = config.n_heads / config.n_kv_heads;

    const LayerWeights& layer = weights.layers.at(layer_idx);
    Tensor& q = state.q;

    Tensor& cache_k = state.cache_ks.at(layer_idx);
    Tensor& cache_v = state.cache_vs.at(layer_idx);
    Tensor k = view_row(cache_k, pos);
    Tensor v = view_row(cache_v, pos);

    // calc q, k, v
    linear(x, layer.wq, q); // q = (dim,)    = (n_heads * head_dim)
    linear(x, layer.wk, k); // k = (kv_dim,) = (n_kv_heads * head_dim)
    linear(x, layer.wv, v);

    // RoPE
    rotary_embedding(q, pos, head_size);
    rotary_embedding(k, pos, head_size);

    const Tensor& attn = state.attn;
    Tensor& context = state.context;
    fill(context, 0.0f);
    const float scale = 1.0f / sqrt(head_size);

#pragma omp parallel for
    for (int32_t h = 0;h < config.n_heads;h++) {
        // float* q_ptr = q.ptr<float>() + h * head_size;
        // const int32_t attn_offset = h * config.max_seq_len;
        // float* attn_ptr = attn.ptr<float>() + attn_offset;
        Tensor q_head = view_1d(q, h * head_size, head_size);
        Tensor attn_head = view_1d(attn, h * config.max_seq_len, config.max_seq_len);
        float* attn_ptr = attn_head.ptr<float>();

        for (int32_t t = 0;t <= pos;t++) {
            // float* k_ptr = cache_k.ptr<float>() + (h / kv_mul) * head_size + t * kv_dim;
            // for (int32_t i = 0;i < head_size;i++) {
            //     sorce += q_ptr[i] * k_ptr[i];
            // }
            Tensor k_t = view_row(cache_k, t);
            float sorce = dot_1d(q_head, view_1d(k_t, (h / kv_mul) * head_size, head_size));

            attn_ptr[t] = sorce * scale;
        }

        // softmax_1d(attn, attn_offset, attn_offset + pos + 1);
        softmax_1d(attn_head, 0, pos + 1);

        // float* c_ptr = context.ptr<float>() + h * head_size;
        Tensor context_head = view_1d(context, h * head_size, head_size);

        for (int32_t t = 0;t <= pos;t++) {
            // float* v_ptr = cache_v.ptr<float>() + (h / kv_mul) * head_size + t * kv_dim;
            // for (int32_t i = 0;i < head_size;i++) {
            //     c_ptr[i] += prob * v_ptr[i];
            // }
            Tensor v_t = view_row(cache_v, t);
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