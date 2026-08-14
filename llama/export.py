import struct  # noqa: I001
from pathlib import Path
import json
import argparse
from collections.abc import Iterator

import torch
from torch import Tensor
from transformers import AutoTokenizer

from model import LLaMA

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAGIC_NUM = 0x70696E65  # pine
VERSION = 2
HEADER_BYTES = 256
TENSOR_ALIGNMENT = 4

MODEL_FILE_TYPE = 0
TOKENIZER_FILE_TYPE = 1


def write_tensor(f, tensor: Tensor, dtype: torch.dtype):
    array = tensor.detach().to(dtype=dtype, device="cpu").contiguous().numpy()
    array.tofile(f)


@torch.inference_mode()
def quantize(tensor: Tensor) -> tuple[Tensor, Tensor]:
    tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    max_vals = tensor.abs().amax(dim=-1)
    scales = (max_vals / 127.0).clamp(min=1e-12)
    values = torch.round(tensor / scales.unsqueeze(-1)).to(dtype=torch.int8)
    return values, scales


def write_quantized_tensor(f, tensor: Tensor):
    values, scales = quantize(tensor)
    write_tensor(f, values, dtype=torch.int8)
    pad = (-f.tell()) % TENSOR_ALIGNMENT
    f.write(b"\0" * pad)
    write_tensor(f, scales, dtype=torch.float32)


def iter_weights(model: LLaMA) -> Iterator[tuple[Tensor, bool]]:
    # token embedding table
    yield model.embed_tokens.weight, False

    # layer weights
    for layer in model.layers:
        # attention weights
        yield layer.self_attn.q_proj.weight, True
        yield layer.self_attn.k_proj.weight, True
        yield layer.self_attn.v_proj.weight, True
        yield layer.self_attn.o_proj.weight, True
        # rmsnorm weigths
        yield layer.input_layernorm.weight, False
        yield layer.post_attention_layernorm.weight, False
        # ffn weigths
        yield layer.mlp.gate_proj.weight, True
        yield layer.mlp.up_proj.weight, True
        yield layer.mlp.down_proj.weight, True

    # header weigths
    yield model.lm_head.weight, True
    # rmsnorm weigths
    yield model.norm.weight, False


def pad_header(f, offset: int):
    pad = offset - f.tell()
    if pad < 0:
        raise ValueError("Header size is greater than header_bytes.")
    f.write(b"\0" * pad)


def export_model(file_path: str | Path, model: LLaMA, quantized: bool):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC_NUM))
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<B", MODEL_FILE_TYPE))
        f.write(struct.pack("<B", quantized))

        cfg = model.cfg
        hidden_dim = model.layers[0].mlp.gate_proj.weight.shape[0]  # ffn hidden dim

        config = struct.pack(
            "<iiiiiifi",
            cfg.dim,
            hidden_dim,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.vocab_size,
            cfg.norm_eps,
            cfg.max_seq_len,
        )
        f.write(config)
        pad_header(f, HEADER_BYTES)

        for tensor, can_quantize in iter_weights(model):
            if quantized and can_quantize:
                write_quantized_tensor(f, tensor)
            else:
                write_tensor(f, tensor, torch.float32)


def export_tokenizer(file_path: str | Path, model_name: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = json.loads(tokenizer.backend_tokenizer.to_str())
    model = data["model"]

    if model["type"] != "BPE":
        # TinyLlama tokenzier type is BPE
        raise ValueError(f"Unsupported tokenizer type: {model['type']}.")
    vocab: dict[str, int] = model["vocab"]
    merges: list[list[str]] = model["merges"]

    vocab_items = sorted(vocab.items(), key=lambda x: x[1])

    with open(file_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC_NUM))
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<B", TOKENIZER_FILE_TYPE))
        f.write(
            struct.pack(
                "<iiiIIII",
                len(vocab),
                len(merges),
                tokenizer.model_max_length,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
            )
        )
        pad_header(f, HEADER_BYTES)

        for token, id in vocab_items:
            token = token.replace("▁", " ")  # replace "▁" to whitespace
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<II", len(token_bytes), id))
            f.write(token_bytes)

        for rank, merge in enumerate(merges):
            assert len(merge) == 2
            left, right = merge[:2]
            left = left.replace("▁", " ")
            right = right.replace("▁", " ")

            left_bytes = left.encode("utf-8")
            right_bytes = right.encode("utf-8")
            f.write(struct.pack("<III", rank, len(left_bytes), len(right_bytes)))
            f.write(left_bytes)
            f.write(right_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Export TinyLlama model & tokenizer binary file."
    )
    parser.add_argument(
        "--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--model_path", type=str, default="./artifacts/model.bin")
    parser.add_argument(
        "--qmodel_path", type=str, default="./artifacts/model-W8A32.bin"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="./artifacts/tokenizer.bin"
    )
    args = parser.parse_args()

    model = LLaMA.from_pretrained(args.model, device_map="cpu")
    export_model(args.model_path, model, quantized=False)
    export_model(args.qmodel_path, model, quantized=True)
    export_tokenizer(args.tokenizer_path, args.model)


if __name__ == "__main__":
    main()
