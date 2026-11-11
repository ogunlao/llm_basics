import math
import torch
import torch.nn as nn
from einops import rearrange, einsum, repeat, reduce
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Bool
from llm_basics.layers import MHSA
from llm_basics.layers import RMSNorm
from llm_basics.layers import SwiGLU
from llm_basics.layers import softmax
from llm_basics.layers import Linear
from llm_basics.layers import RotaryPositionalEmbedding
from llm_basics.layers import Embedding


class SwitchBlock(nn.Module):
    def __init__(self, num_experts: int, capacity_factor: float | None, d_model: int, num_heads: int, d_ff: int, rope_emb: nn.Module | None, jitter_noise: float = 1e-2, device: torch.device | None=None, dtype: torch.dtype | None=None, context_length: int | None=None, use_qk_norm: bool=False,):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mhsa = MHSA(d_model, num_heads, rope_emb, device=device, dtype=dtype, context_length=context_length, use_qk_norm=use_qk_norm)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff, device, dtype) for _ in range(num_experts)])
        self.router = Linear(d_model, num_experts, device=device, dtype=torch.float32)
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise

    def switch_layer(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        in_dtype = x.dtype
        batch, seq_len, d_model = x.size()

        x = rearrange(x, "batch seq_len d_model -> (batch seq_len) d_model")
        
        num_tokens = batch * seq_len
        capacity = None
        if self.capacity_factor is not None:
            capacity = math.ceil(num_tokens * self.capacity_factor / len(self.experts))

        x_router = x.to(torch.float32)
        if self.training and self.jitter_noise:
            # Add some noise for exploration across experts
            random_noise = torch.zeros_like(x_router).uniform_(1 - self.jitter_noise, 1 + self.jitter_noise)
            x_router = x_router * random_noise.to(x_router.device)
        # Router logits and selection in float32 for numerical stability, also under fp16 autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = self.router(x_router)

        router_probs = softmax(router_logits, dim=-1)
        
        routing_max_prob, routing_indices = torch.max(router_probs, dim=-1)
        routing_max_prob = routing_max_prob.to(in_dtype)

        output = torch.zeros_like(x)

        # for auxilliary loss computation
        no_tokens_per_experts = torch.zeros(len(self.experts), device=x.device)
        router_probs_per_expert = torch.zeros(len(self.experts), device=x.device)

        # Choosing to route to experts with confidence-based truncation
        for moe_idx, expert in enumerate(self.experts):
            expert_mask = (routing_indices == moe_idx)
            if not expert_mask.any(): # not token in this batch choose this expert
                continue
            expert_indices = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)
            probs = routing_max_prob[expert_indices]
            if self.training and self.capacity_factor is not None and len(expert_indices) > capacity:
                # Route by highest confidence scores on routing probability
                probs = routing_max_prob[expert_indices]
                sorted_order = torch.argsort(probs, descending=True)

                kept_indices = expert_indices[sorted_order[:capacity]]
                overflow_indices = expert_indices[sorted_order[capacity:]]

                # Process kept tokens and scale by gate prob
                expert_out = expert(x[kept_indices])
                output[kept_indices] = expert_out * routing_max_prob[kept_indices].unsqueeze(-1)
            else:
                expert_out = expert(x[expert_indices])
                output[expert_indices] = expert_out * routing_max_prob[expert_indices].unsqueeze(-1)

            no_tokens_per_experts[moe_idx] = len(expert_indices)
            router_probs_per_expert[moe_idx] = router_probs[..., moe_idx].sum()

        output = rearrange(output, "(batch seq_len) d_model -> batch seq_len d_model", batch=batch)
        aux_loss = self.load_balance_loss(no_tokens_per_experts, router_probs_per_expert, num_tokens)

        return output, aux_loss

    def load_balance_loss(self, no_tokens_per_experts: Float[torch.Tensor, "exp_len"], router_probs_per_expert: Float[torch.Tensor, "exp_len"], num_tokens: int) -> Float[torch.Tensor, ""]:

        no_tokens_per_experts = no_tokens_per_experts / num_tokens
        router_probs_per_expert = router_probs_per_expert / num_tokens

        aux_loss = einsum(no_tokens_per_experts, router_probs_per_expert, "exp_len, exp_len ->" )
        aux_loss *= len(self.experts)

        return aux_loss

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"], token_pos=None, mask=None, use_cache: bool=False) -> Float[torch.Tensor, "batch seq_len d_model"]:
        residual = x
        x = self.rms_norm1(x)
        x = self.mhsa(x, token_pos, mask, use_cache=use_cache)
        x = x + residual
        residual = x
        x = self.rms_norm2(x)
        x, aux_loss = self.switch_layer(x)
        x = residual + x

        return x, aux_loss


class SwitchTransformerLM(nn.Module):
    def __init__(self, num_experts: int, capacity_factor: float | None, vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, context_length: int, theta: float | None=None, jitter_noise: float = 1e-2, device: torch.device | None=None, dtype: torch.dtype | None=None, use_qk_norm: bool=False,):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.rope_emb = None
        if theta is not None:
            self.rope_emb = RotaryPositionalEmbedding(theta, d_model//num_heads, context_length, device)
        self.layers = nn.ModuleList(SwitchBlock(num_experts, capacity_factor, d_model, num_heads, d_ff, self.rope_emb, jitter_noise, device, dtype, context_length, use_qk_norm,) for _ in range(num_layers))
        self.rms_norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, vocab_size, device, dtype)

    def forward(self, input_tokens: Int[torch.Tensor, "... seq_len"], token_pos=None, mask=None, use_cache: bool=False) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.emb(input_tokens)
        aux_losses = torch.zeros(1, device=input_tokens.device)
        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x, token_pos, mask, use_cache=use_cache)
            aux_losses += aux_loss
        x = self.rms_norm_final(x)
        x = self.out_proj(x) # output: logits
        return x, aux_losses / len(self.layers)

    @torch.no_grad()
    def reset_kv_cache(self):
        for layer in self.layers:
            layer.mhsa.reset_cache()


if __name__ == "__main__":
    bs, seq_len, d_model = 2, 7, 8
    vocab_size = 100
    device = "mps"
    x = torch.rand((bs, seq_len, d_model))
    out_dim = 10
    num_heads = 2
    theta = 10000
    d_k = d_model // num_heads
    context_len = 500
    num_experts = 2
    capacity_factor = 1.2
    jitter_noise = 1e-2

    d_ff = math.ceil(((8 / 3) * d_model) / 64) * 64
    rope_emb = RotaryPositionalEmbedding(theta, d_k, context_len, device)
    t_block = SwitchBlock(num_experts, capacity_factor, d_model, num_heads, d_ff, rope_emb, jitter_noise, device=device)
    y, _ = t_block(x.to(device))
    assert y.shape == x.shape

    a = torch.randint(high=99, size=(bs, seq_len)).to(torch.long)
    xformer = SwitchTransformerLM(num_experts, capacity_factor, vocab_size, 2, d_model, num_heads, d_ff, context_len, theta, jitter_noise, device=device)
    y, _ = xformer(a.to(device))
    assert tuple(y.shape) == tuple((*a.shape, vocab_size))