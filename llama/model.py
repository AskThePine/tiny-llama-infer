from dataclasses import dataclass

import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    dim: int = 2048
    n_layers: int = 22
    n_heads: int = 32
    n_kv_heads: int = 4
    vocab_size: int = 32000
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()

        # theta_i = 1 / base^{2i/d}
        inv_freq = 1 / base ** (torch.arange(0, dim, step=2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: Tensor, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        # position_ids: (B, T)

        position_ids_expanded = position_ids.unsqueeze(-1).float()  # (B, T, 1)
        inv_freq_expanded = self.inv_freq.unsqueeze(-2).float()  # (B, 1, dim/2)
        freqs = position_ids_expanded @ inv_freq_expanded  # (B, T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, T, dim)

        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: Tensor) -> Tensor:
    # x: (nh, T, hs)
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)  # (B, 1, T, dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    # x: (B, nkvh, T, hdim)
    # B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    # (B, n_rep * nkvh, T, hdim)
    return torch.repeat_interleave(x, repeats=n_rep, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        assert cfg.dim % cfg.n_heads == 0
        self.head_dim = cfg.dim // cfg.n_heads
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)

        self.cache_k = torch.zeros(
            cfg.max_batch_size, cfg.n_kv_heads, cfg.max_seq_len, self.head_dim
        )
        self.cache_v = torch.zeros(
            cfg.max_batch_size, cfg.n_kv_heads, cfg.max_seq_len, self.head_dim
        )

    def forward(
        self, x: Tensor, pos_embd: tuple[Tensor, Tensor], start_pos: int, mask: Tensor
    ) -> Tensor:
        B, T, dim = x.shape
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # (B, nh, T, hdim)
        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        # (B, nkvh, T, hdim)
        k = k.view(B, T, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)

        # apply RoPE
        cos, sin = pos_embd
        q, k = apply_rotary_pos_emb(
            q, k, cos, sin
        )  # q: (B, nh, T, hdim) k: (B, nkvh, T, hdim)

        # KV Cache
        self.cache_k = self.cache_k.to(k)
        self.cache_v = self.cache_v.to(v)

        self.cache_k[:B, :, start_pos : start_pos + T, :] = k
        self.cache_v[:B, :, start_pos : start_pos + T, :] = v

        keys = self.cache_k[:B, :, : start_pos + T, :]  # (B, nkvh, cache_len + T, hdim)
        values = self.cache_v[:B, :, : start_pos + T, :]
        querys = q  # (B, nh, T, hdim)
        keys = repeat_kv(keys, self.n_rep)  # (B, nh, cache_len + T, hdim)
        values = repeat_kv(values, self.n_rep)

        # (B, nh, T, cache_len + T)
        scores = (querys @ keys.transpose(-1, -2)) * (self.head_dim**-0.5)
        if mask is not None:
            # mask: (T, cache_len + T)
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        out = scores @ values  # (B, nh, T, hdim)
        out = out.transpose(1, 2).contiguous().view(B, T, dim)  # (B, T, dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x_gated = self.gate_proj(x)  # (B, T, hdim)
        h = F.silu(x_gated) * self.up_proj(x)  # (B, T, hdim) * (B, T, hdim)
        out = self.down_proj(h)  # (B, T, dim)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.self_attn = Attention(cfg)
        self.mlp = FeedForward(
            dim=cfg.dim, hidden_dim=4 * cfg.dim, multiple_of=cfg.multiple_of
        )
        self.input_layernorm = RMSNorm(dim=cfg.dim, eps=cfg.norm_eps)
        self.post_attention_layernorm = RMSNorm(dim=cfg.dim, eps=cfg.norm_eps)

    def forward(
        self, x: Tensor, pos_embd: tuple[Tensor, Tensor], start_pos: int, mask: Tensor
    ) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), pos_embd, start_pos, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LLaMA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

        # self.lm_head.weight = self.embed_tokens.weight

        assert cfg.dim % cfg.n_heads == 0
        self.rotary_emb = RotaryEmbedding(cfg.dim // cfg.n_heads)

    def forward(self, ids: Tensor, start_pos: int) -> Tensor:
        B, T = ids.shape
        pos_embd = self.rotary_emb(
            torch.arange(start_pos, start_pos + T, device=ids.device)
            .unsqueeze(0)
            .expand(B, -1),
            dtype=self.embed_tokens.weight.dtype,
        )  # tuple (B, T, dim)

        h = self.embed_tokens(ids)
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=ids.device)
            mask = torch.triu(mask, diagonal=1)  # (T, T)

            mask = torch.hstack(
                (torch.zeros(T, start_pos, device=ids.device), mask)
            ).type_as(h)  # (T, start_pos + T)

        for layer in self.layers:
            h = layer(h, pos_embd, start_pos, mask)
        h = self.norm(h)
        out = self.lm_head(h)
        return out

    @classmethod
    def from_pretrained(
        cls, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cpu"
    ):
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, device_map=device_map
        )
        hf_cfg = hf_model.config
        hf_sd = hf_model.state_dict()

        cfg = ModelConfig(
            dim=hf_cfg.hidden_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_heads=hf_cfg.num_attention_heads,
            n_kv_heads=getattr(
                hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads
            ),
            vocab_size=hf_cfg.vocab_size,
            norm_eps=hf_cfg.rms_norm_eps,
            max_seq_len=hf_cfg.max_position_embeddings,
        )

        model = cls(cfg)
        sd = model.state_dict()
        assert len(hf_sd) == len(sd), f"mismatched keys: {len(hf_sd)} != {len(sd)}"
        sd = {}
        for k, v in hf_sd.items():
            sd[k.removeprefix("model.")] = v

        model.load_state_dict(sd)
        return model
