import torch
from torch import Tensor

from llama.model import LLaMA


class Generate:
    def __init__(self, model: LLaMA, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> list[list[int]]:
        cfg = self.model.cfg
        B = len(prompt_tokens)
        assert B <= cfg.max_batch_size

        min_seq_len = min(len(ids) for ids in prompt_tokens)
        max_seq_len = max(len(ids) for ids in prompt_tokens)
        assert max_seq_len <= cfg.max_seq_len
        total_seq_len = min(cfg.max_seq_len, max_seq_len + max_gen_len)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        # ids: (B, L), L = total_seq_len
        ids = torch.full((B, total_seq_len), pad_id, dtype=torch.long, device=device)
        for i, tokens in enumerate(prompt_tokens):
            ids[i, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )

        mask: Tensor = ids != pad_id  # (B, L)
        eos_reached = torch.tensor([False] * B, device=device)  # (B,)

        prev_pos = 0
        for cur_pos in range(min_seq_len, total_seq_len):
            with torch.no_grad():
                logits = self.model.forward(
                    ids[:, prev_pos:cur_pos], prev_pos
                )  # (B, T, vocab_size), T = min_seq_len or 1
            if temperature > 0:
                probs = torch.softmax(
                    logits[:, -1, :] / temperature, dim=-1
                )  # (B, dim)
                next_token = sample_top_p(probs, top_p)  # (B, 1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)

            next_token = next_token.reshape(-1)  # (B,)
            next_token = torch.where(mask[:, cur_pos], ids[:, cur_pos], next_token)
            ids[:, cur_pos] = next_token

            eos_reached |= (~mask[:, cur_pos]) & (next_token == eos_id)
            prev_pos = cur_pos
            if eos_reached.all().item():
                break

        out_tokens = []
        for i, tokens in enumerate(ids.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            end = min(total_seq_len, len(prompt_tokens[i]) + max_gen_len)
            tokens = tokens[start:end]
            if eos_id in tokens:
                eos_idx = tokens.index(eos_id)
                tokens = tokens[:eos_idx]
            out_tokens.append(tokens)

        return out_tokens  # (B, ?)

    def text_completion(
        self,
        prompts: str | list[str],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> str | list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt_tokens: list[list[int]] = [self.tokenizer.encode(s) for s in prompts]
        out_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            device=device,
        )
        return self.tokenizer.decode(out_tokens, skip_special_tokens=True)


def sample_top_p(probs: Tensor, top_p: float) -> Tensor:
    probs_sort, probs_idx = torch.sort(
        probs, dim=-1, descending=True
    )  # (B, vocab_size)
    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_cumsum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort /= torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token_id = torch.multinomial(probs_sort, num_samples=1)  # (B, 1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token_id)
    return next_token
