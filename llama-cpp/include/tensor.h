#pragma once
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <array>
#include <initializer_list>

constexpr int32_t MAX_DIMS = 4;

enum DType {
    Float32,
    Int8
};

constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::Float32:
        return sizeof(float);
    case DType::Int8:
        return sizeof(int8_t);
    }
    return 0;
}

struct Tensor {
    void* data;
    DType dtype;
    int32_t ndim; // num dim
    std::array<int32_t, MAX_DIMS> shape;
    std::array<int64_t, MAX_DIMS> stride;

    template<typename T>
    T* ptr() {
        return static_cast<T*>(data);
    }

    template<typename T>
    const T* ptr() const {
        return static_cast<T*>(data);
    }

    int64_t numel() const {
        int64_t res = 1;
        for (int i = 0;i < ndim;i++) {
            res *= shape[i];
        }
        return res;
    }

    int64_t nbytes() const {
        return numel() * dtype_size(dtype);
    }

    bool is_dtype(DType dtype) {
        return this->dtype == dtype;
    }
};

inline Tensor make_tensor(void* data, DType dtype, int32_t ndim, const int32_t* shape) {
    assert(1 <= ndim && ndim <= MAX_DIMS);
    Tensor t{};
    t.data = data;
    t.dtype = dtype;
    t.ndim = ndim;

    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        assert(shape[i] > 0);
        t.shape[i] = shape[i];
        t.stride[i] = stride;
        stride *= shape[i];
    }
    return t;
}

inline Tensor make_tensor(void* data, DType dtype, const std::initializer_list<int32_t> shape) {
    int32_t ndim = shape.size();
    assert(1 <= ndim && ndim <= MAX_DIMS);
    return make_tensor(data, dtype, ndim, shape.begin());
}

inline Tensor make_view(void* data, DType dtype, const std::initializer_list<int32_t> shape) {
    // Currently only supports continuous tensor
    return make_tensor(data, dtype, shape);
}

inline Tensor view_1d(const Tensor& x, int64_t start, int32_t length) {
    assert(x.ndim == 1);
    assert(start >= 0);
    assert(0 < length && start <= x.shape[0] - length);

    uint8_t* ptr = static_cast<uint8_t*>(x.data) + (size_t)start * x.stride[0] * dtype_size(x.dtype);
    return make_view(ptr, x.dtype, { length });
}

inline Tensor view_row(const Tensor& x, int32_t row) {
    assert(x.ndim == 2);
    int32_t rows = x.shape[0];
    int32_t cols = x.shape[1];
    assert(0 <= row && row < rows);

    uint8_t* ptr = static_cast<uint8_t*>(x.data) + (size_t)row * x.stride[0] * dtype_size(x.dtype);
    return make_view(ptr, x.dtype, { cols });
}

std::ostream& operator<<(std::ostream& os, const Tensor& t);

void fill(Tensor& x, const float val = 0.0f);

void copy(const Tensor& x, Tensor& out);

void add_1d(const Tensor& x, Tensor& out);

float dot_1d(const Tensor& x, const Tensor& y);

void mul_1d(const Tensor& x, const Tensor& y, Tensor& out);

void div_1d(Tensor& x, const float val);

void axpy_1d(const Tensor& x, const float val, Tensor& out);

void matvec(const Tensor& x, const Tensor& w, Tensor& out);

void matmul_2d(const Tensor& lhs, const Tensor& rhs, Tensor& out);

void softmax_1d(Tensor& x, int32_t start, int32_t end);

void softmax_1d(Tensor& x);

void silu_1d(Tensor& x);

void qmatvec(const Tensor& x, const Tensor& qw, const Tensor& scales, Tensor& out);
