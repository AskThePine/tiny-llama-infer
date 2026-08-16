# TinyLlama Inference Learning Project

---
This is a TinyLlama inference project for learning, with both a Python reference implementation
and a C++ CPU inference implementation. It covers single-token Transformer inference, GQA, RoPE,
KV Cache, BPE tokenization, model export, memory-mapped loading, and W8A32 weight quantization.

The current implementation primarily targets `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. It uses a
custom binary format and cannot read Hugging Face's original checkpoint files directly.

The overall learning direction, C++ inference flow, model configuration, and weight-loading design
are heavily inspired by [`karpathy/llama2.c`](https://github.com/karpathy/llama2.c), with
adaptations and extensions for TinyLlama's GQA, BPE tokenizer, `mmap`, and W8A32 quantization
requirements.

---

## Project Structure

```text
.
├── llama/
│   ├── model.py       # Python Transformer reference implementation 
│   │                    and Hugging Face weight conversion
│   ├── generate.py    # Python generation and top-p sampling reference code
│   └── export.py      # Export models, tokenizer, and W8A32 weights
├── llama-cpp/
│   ├── include/       # C++ headers and data structures
│   ├── src/           # Loader, model, tokenizer, sampler, and generator
│   └── CMakeLists.txt # CMake build configuration
└── requirements.txt   # Python dependencies
```

## Implementation Overview

The Python side provides the reference implementation and model conversion tools:

- Loads TinyLlama weights from Hugging Face;
- Implements RMSNorm, RoPE, GQA, KV Cache, and the SwiGLU FFN;
- Exports model weights and the tokenizer to a binary format readable by the C++ loader;
- Exports both Float32 and W8A32 quantized models.

The C++ side provides CPU inference:

- Implements single-token incremental inference with C++17;
- Uses `mmap` to map model files on POSIX systems, allowing Tensors to reference the mapped region directly;
- Implements a BPE tokenizer with byte fallback;
- Supports Float32 and W8A32 weights;
- Supports greedy sampling, temperature, top-p sampling, and fixed random seeds.

## Requirements

### Python Dependencies

Run this command from the project root:

```bash
python -m pip install -r requirements.txt
```

On the first export, Transformers downloads the model from Hugging Face. A locally cached model
can also be used. The export process generates a Float32 model, a W8A32 model, and a tokenizer
file, so it may require a significant amount of disk space.

### C++ Dependencies

The C++ implementation requires:

- CMake 3.15 or later;
- A compiler with C++17 support;
- OpenMP;
- The ICU Unicode library.

On Ubuntu/Debian, install the dependencies with:

```bash
sudo apt install build-essential cmake libicu-dev
```

## Quick Start

Run all commands below from the project root.

### 1. Export the Model and Tokenizer

```bash
python llama/export.py
```

The default output files are:

```text
artifacts/model.bin         # Float32 model
artifacts/model-W8A32.bin   # W8A32 model
artifacts/tokenizer.bin     # tokenizer
```

You can also specify the model name and output paths:

```bash
python3 llama/export.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --model_path artifacts/model.bin \
    --qmodel_path artifacts/model-W8A32.bin \
    --tokenizer_path artifacts/tokenizer.bin
```

### 2. Build the C++ Inference Program

```bash
cmake -S llama-cpp -B llama-cpp/build
cmake --build llama-cpp/build -j
```

The executable will be located at:

```text
llama-cpp/build/llama_cpp
```

### 3. Generate Text with the Float32 Model

```bash
./llama-cpp/build/llama_cpp \
    -ckpt artifacts/model.bin \
    -tokenizer artifacts/tokenizer.bin \
    -prompt "The capital of France is" \
    -len 64 \
    -temperature 0.5 \
    -topp 0.9 \
    -seed 42 \
    -echo
```

### 4. Generate Text with the W8A32 Model

Only the checkpoint path needs to be changed:

```bash
./llama-cpp/build/llama_cpp \
    -ckpt artifacts/model-W8A32.bin \
    -tokenizer artifacts/tokenizer.bin \
    -prompt "The capital of France is" \
    -len 64 \
    -temperature 0.7 \
    -topp 0.9 \
    -seed 42
```

## Command-Line Arguments

| Argument | Alias | Default | Description |
| --- | --- | --- | --- |
| `-ckpt` | `-checkpoint` | none | Model file path |
| `-tokenizer` | `-t` | none | Tokenizer file path |
| `-prompt` | `-p` | none | Input prompt; must not be empty |
| `-len` | `-l` | `64` | Maximum number of generated tokens |
| `-temperature` | `-T` | `1.0` | Temperature; set to `0` to use greedy sampling |
| `-topp` | `-P` | `0.9` | Cumulative probability for nucleus sampling, in `[0, 1]` |
| `-seed` | `-s` | `42` | Random sampling seed |
| `-echo` | `-e` | disabled | Include the input prompt in the output |

When `temperature=0`, the program directly selects the token with the highest logit, so `top-p` and `seed` have no effect. The generation length is limited by the model's `max_seq_len`.

## Model File Format

The project uses a custom binary format, currently at version `2`. All integers and floating-point
values use little-endian encoding, and the file header is padded to 256 bytes.

### Model File

The model header contains:

```text
magic number
version
file type
flags
model config
padding
```

The lowest bit of `flags` indicates whether the model is quantized:

- `0`: Float32 model;
- `1`: W8A32 model.

### Tokenizer File

The tokenizer header contains:

```text
magic number
version
file type
tokenizer config
padding
```

Vocabulary entries and merge rules are stored after the header. During loading, merge-rule strings
are resolved to token IDs, so encoding does not repeatedly concatenate and look up token strings.

### W8A32 Weights

W8A32 means that weights use Int8 while activations remain Float32. Only linear-layer weights are
quantized; embeddings, RMSNorm weights, and other non-linear-layer weights remain Float32.

Each linear layer is quantized symmetrically per row:

```text
scale[row] = max(abs(row)) / 127
values[row] = round(row / scale[row])
```

A quantized linear layer is stored in the following order:

```text
int8 values -> 4-byte alignment padding -> float32 scales
```

### Weight Loading

Linux and other POSIX systems use `mmap` to map the model file, allowing weight Tensors to reference
the mapped region directly without copying the entire model again. Windows uses a file buffer as a
compatibility implementation.

Runtime buffers and the KV Cache are allocated separately from the model mapping. The loader checks
the file header, configuration, derived Tensor data ranges, and the final file offset.

The model and tokenizer should come from the same model configuration. The program currently checks
only whether `vocab_size` and `context length` match; it does not guarantee that the complete
vocabulary and merge rules are identical.

## Current Scope and Limitations

- The implementation primarily targets the TinyLlama LLaMA architecture and its BPE tokenizer.
- C++ inference is CPU-only and does not include CUDA or another GPU backend.
- The current C++ generator performs single-sequence, token-by-token inference and does not provide a batch inference interface.
- The custom binary format is not compatible with original Hugging Face checkpoints; the export script must be run first.
- W8A32 quantizes only linear-layer weights and is not full end-to-end Int8 inference.
- Model files in `artifacts/` are usually large and are ignored by Git by default.
- The Python code is primarily for reference and export; the C++ command-line program is the main end-to-end entry point.

## Acknowledgments

- [llama2.c](https://github.com/karpathy/llama2.c)

- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
