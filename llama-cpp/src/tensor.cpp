#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

    void is_valid(const Tensor& x) {
        assert(x.data != nullptr);
        assert(1 <= x.ndim && x.ndim <= MAX_DIMS);
        for (int32_t i = 0;i < x.ndim;i++) {
            assert(x.shape[i] >= 1 && x.stride[i] >= 1);
        }
    }

    bool is_contiguous(const Tensor& x) {
        int64_t expected_stride = 1;
        for (int32_t i = x.ndim - 1; i >= 0; --i) {
            if (x.stride[i] != expected_stride) {
                return false;
            }
            expected_stride *= x.shape[i];
        }
        return true;
    }

    void assert_contiguous(const Tensor& x) {
        is_valid(x);
        assert(is_contiguous(x));
    }

    void assert_contiguous(const std::initializer_list<Tensor>& xs) {
        for (const Tensor& x : xs) {
            assert_contiguous(x);
        }
    }

    void assert_ndim(const int32_t ndim, const Tensor& x) {
        assert(x.ndim == ndim);
    }

    void assert_ndim(const int32_t ndim, const std::initializer_list<Tensor>& xs) {
        for (const Tensor& x : xs) {
            assert_ndim(ndim, x);
        }
    }

    void assert_dtype(const DType dtype, const Tensor& x) {
        assert(x.dtype == dtype);
    }

    void assert_dtype(const DType dtype, const std::initializer_list<Tensor>& xs) {
        for (const Tensor& x : xs) {
            assert_dtype(dtype, x);
        }
    }

}  // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(ndim=" << t.ndim << ", shape=[";
    for (int32_t i = 0;i < t.ndim;i++) {
        if (i) os << ", ";
        os << t.shape[i];
    }
    os << "], stride=[";
    for (int32_t i = 0;i < t.ndim;i++) {
        if (i) os << ", ";
        os << t.stride[i];
    }
    os << "], numel=" << t.numel() << ", data=" << t.data << ")";
    return os;
}

void fill(Tensor& x, const float val) {
    assert_contiguous(x);
    assert_dtype(DType::Float32, x);
    float* data = x.ptr<float>();
    std::fill_n(data, x.numel(), val);
}

void copy(const Tensor& x, Tensor& out) {
    assert_contiguous({ x,out });
    assert(x.dtype == out.dtype);

    assert(x.ndim == out.ndim);
    for (int32_t i = 0;i < x.ndim;i++) {
        assert(x.shape[i] == out.shape[i]);
    }

    const float* x_ptr = x.ptr<float>();
    float* o_ptr = out.ptr<float>();
    std::memmove(o_ptr, x_ptr, x.nbytes());
}

void add_1d(const Tensor& x, Tensor& out) {
    assert_contiguous({ x,out });
    assert_ndim(1, { x,out });
    assert_dtype(DType::Float32, { x,out });

    const int32_t d = out.shape[0];
    const float* x_ptr = x.ptr<float>();
    float* o_ptr = out.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        o_ptr[i] += x_ptr[i];
    }
}

float dot_1d(const Tensor& x, const Tensor& y) {
    assert_contiguous({ x,y });
    assert_ndim(1, { x,y });
    assert_dtype(DType::Float32, { x,y });

    const int32_t d = x.shape[0];
    assert(d == y.shape[0]);
    const float* x_ptr = x.ptr<float>();
    const float* y_ptr = y.ptr<float>();

    float res = 0.0f;
    for (int32_t i = 0;i < d;i++) {
        res += x_ptr[i] * y_ptr[i];
    }
    return res;
}

void mul_1d(const Tensor& x, const Tensor& y, Tensor& out) {
    // x (d,) * y (d,) = out (d,)
    assert_contiguous({ x,y,out });
    assert_ndim(1, { x,y,out });
    assert_dtype(DType::Float32, { x,y,out });

    const int32_t d = out.shape[0];
    assert(d == x.shape[0] && d == y.shape[0]);

    const float* x_ptr = x.ptr<float>();
    const float* y_ptr = y.ptr<float>();
    float* o_ptr = out.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        o_ptr[i] = x_ptr[i] * y_ptr[i];
    }
}

void div_1d(Tensor& x, const float val) {
    assert_contiguous(x);
    assert_ndim(1, x);
    assert_dtype(DType::Float32, x);
    assert(val != 0.0f);

    const int32_t d = x.shape[0];
    float* x_ptr = x.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        x_ptr[i] /= val;
    }
}

void axpy_1d(const Tensor& x, const float val, Tensor& out) {
    assert_contiguous({ x,out });
    assert_ndim(1, { x,out });
    assert_dtype(DType::Float32, { x,out });

    const int32_t d = out.shape[0];
    assert(d == x.shape[0]);

    const float* x_ptr = x.ptr<float>();
    float* o_ptr = out.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        o_ptr[i] += val * x_ptr[i];
    }
}

void matvec(const Tensor& x, const Tensor& w, Tensor& out) {
    // W (d, n) @ x (n,) = out (d,)
    assert_contiguous({ x,w,out });
    assert_ndim(1, { x,out });
    assert_ndim(2, w);
    assert_dtype(DType::Float32, { x,w,out });
    assert(out.data != x.data);
    assert(out.data != w.data);

    const int32_t d = w.shape[0];
    const int32_t n = w.shape[1];
    assert(n == x.shape[0]);
    assert(d == out.shape[0]);

    const float* x_ptr = x.ptr<float>();
    const float* w_ptr = w.ptr<float>();
    float* o_ptr = out.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        float val{};
        for (int32_t j = 0;j < n;j++) {
            val += w_ptr[i * n + j] * x_ptr[j];
        }
        o_ptr[i] = val;
    }
}

void matmul_2d(const Tensor& lhs, const Tensor& rhs, Tensor& out) {
    // lhs: (N, K)
    // rhs: (K, M)
    // out: (N, M)
    assert_contiguous({ lhs,rhs,out });
    assert_ndim(2, { lhs,rhs,out });
    assert_dtype(DType::Float32, { lhs,rhs,out });
    assert(out.data != lhs.data && out.data != rhs.data);

    assert(lhs.shape[1] == rhs.shape[0]);
    const int32_t N = lhs.shape[0];
    const int32_t K = lhs.shape[1];
    const int32_t M = rhs.shape[1];
    assert(out.shape[0] == N && out.shape[1] == M);

    const float* l_ptr = lhs.ptr<float>();
    const float* r_ptr = rhs.ptr<float>();
    fill(out, 0.0f);
    float* o_ptr = out.ptr<float>();

    for (int32_t i = 0;i < N;i++) {
        int64_t o_offset = (int64_t)i * M;
        for (int32_t k = 0;k < K;k++) {
            int64_t l_idx = (int64_t)i * K + k;
            int64_t r_offset = (int64_t)k * M;
            for (int32_t j = 0;j < M;j++) {
                o_ptr[o_offset + j] += l_ptr[l_idx] * r_ptr[r_offset + j];
            }
        }
    }
}

void softmax_1d(Tensor& x, int32_t start, int32_t end) {
    assert_contiguous(x);
    assert_ndim(1, x);
    assert_dtype(DType::Float32, x);

    int32_t d = x.shape[0];
    assert(0 <= start && end <= d);

    float* x_ptr = x.ptr<float>();
    float max_val = x_ptr[start];
    for (int32_t i = start;i < end;i++) {
        if (max_val < x_ptr[i]) {
            max_val = x_ptr[i];
        }
    }

    float sum{};
    for (int32_t i = start;i < end;i++) {
        x_ptr[i] = std::exp(x_ptr[i] - max_val);
        sum += x_ptr[i];
    }
    for (int32_t i = start;i < end;i++) {
        x_ptr[i] /= sum;
    }
}

void softmax_1d(Tensor& x) {
    assert_contiguous(x);
    assert_ndim(1, x);

    softmax_1d(x, 0, x.shape[0]);
}

void silu_1d(Tensor& x) {
    assert_contiguous(x);
    assert_ndim(1, x);
    assert_dtype(DType::Float32, x);

    int32_t d = x.shape[0];
    float* x_ptr = x.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        x_ptr[i] = x_ptr[i] / (1.0f + std::exp(-x_ptr[i]));
    }
}

void sigmoid_1d(Tensor& x) {
    assert_contiguous(x);
    assert_ndim(1, x);
    assert_dtype(DType::Float32, x);

    int32_t d = x.shape[0];
    float* x_ptr = x.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        x_ptr[i] = 1.0f / (1.0f + std::exp(-x_ptr[i]));
    }
}

void qmatvec(const Tensor& x, const Tensor& qw, const Tensor& scales, Tensor& out) {
    // W (d, n) @ x (n,) = out (d,)
    assert_contiguous({ x,qw,scales,out });
    assert_ndim(1, { x,scales,out });
    assert_ndim(2, qw);
    assert_dtype(DType::Float32, { x,scales,out });
    assert_dtype(DType::Int8, qw);
    assert(out.data != x.data);
    assert(out.data != qw.data);
    assert(out.data != scales.data);

    const int32_t d = qw.shape[0];
    const int32_t n = qw.shape[1];
    assert(n == x.shape[0]);
    assert(d == out.shape[0] && d == scales.shape[0]);

    const float* x_ptr = x.ptr<float>();
    const int8_t* w_ptr = qw.ptr<int8_t>();
    const float* s_ptr = scales.ptr<float>();
    float* o_ptr = out.ptr<float>();

#pragma omp parallel for
    for (int32_t i = 0;i < d;i++) {
        float val{};
        for (int32_t j = 0;j < n;j++) {
            val += w_ptr[i * n + j] * x_ptr[j];
        }
        o_ptr[i] = val * s_ptr[i];
    }
}