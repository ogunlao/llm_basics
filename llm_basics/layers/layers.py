import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Bool


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None) -> None:
        """Construct a linear transformation module.
        Args:
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.input_features = in_features
        self.output_features = out_features
        self.weights = nn.Parameter(
            data=torch.empty((out_features, in_features), dtype=dtype, device=device), requires_grad=True)
        self._init_layer()

    def forward(self, x: Float[torch.Tensor, "... input_dim"]) -> Float[torch.Tensor, "... output_dim"]:
        """Apply the linear transformation to the input."""
        return einsum(x, self.weights, "... input_dim, output_dim input_dim -> ... output_dim")

    @torch.no_grad()
    def _init_layer(self):
        rho_squared = 2 / (self.input_features + self.output_features)
        rho = rho_squared ** (0.5)
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=rho, a=-3*rho, b=3*rho)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            data=torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device), requires_grad=True
            )
        self._init_layer()

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weights[token_ids]
        
    def _init_layer(self):
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3, b=3)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        """Construct the Root Mean Square Layer Normalization module."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(data=torch.ones((d_model), dtype=dtype, device=device, requires_grad=True), requires_grad=True)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # x e.g. (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared_x = x ** 2
        sum_squared = einsum(squared_x, "... d_model -> ...")
        sum_squared = sum_squared / self.d_model + self.eps
        rms = sum_squared ** 0.5
        rms = rearrange(rms, "... -> ... 1")
        x = x * self.g / rms
        return x.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        x = self.silu(self.w1(x)) * self.w3(x)
        return self.w2(x)

class RotaryPositionalEmbedding(nn.Module):
    # Implementation help from Youtube tutorial: https://www.youtube.com/watch?v=Kv90HQY9lZA
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0, "d_k must be divisible by 2"
        self.precompute_theta_pos_freq(theta, d_k, max_seq_len, device)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Float[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = rearrange(x, "... (half_d_k two) -> ... half_d_k two", two=2)
        x_complex = torch.view_as_complex(x) # "... d_k/2 2 -> ... d_k/2"
        freqs_complex = self.freqs_complex[token_positions] # "max_seq_len d_k/2 -> seq_len d_k/2"
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = rearrange(x_out, "... half_d_k two -> ... (half_d_k two)")
        return x_out.to(in_dtype)

    def precompute_theta_pos_freq(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        seq_pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # Shape: max_seq_len
        theta_pow = torch.arange(0, d_k, 2, dtype=torch.float32, device=device) # Shape: d_k / 2
        theta_pow /= d_k
        thetas = 1.0 / (theta ** theta_pow) # Shape: d_k / 2
        freqs = einsum(seq_pos, thetas, "max_seq_len, k -> max_seq_len k")
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer("freqs_complex", freqs_complex, persistent=False)


def softmax(x: Float[torch.Tensor, "..."], dim: int, all_dim_masked: bool = False):
    in_dtyp = x.dtype
    x = x.to(torch.float32)
    if all_dim_masked:
        # Detect where every element along the dimension is -inf
        is_all_neginf = torch.all(torch.isneginf(x), dim=dim, keepdim=True)
    
        x = torch.where(is_all_neginf, torch.zeros_like(x), x)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_new = x - x_max
    exp_x = torch.exp(x_new)
    sum_x = exp_x.sum(dim=dim, keepdim=True)
    probs = exp_x / sum_x
    if all_dim_masked:
        probs = torch.where(is_all_neginf, torch.zeros_like(probs), probs)
    return probs.to(in_dtyp)


def scaled_dot_product_attention(
        keys: Float[torch.Tensor, "batch_size ... seq_len_k d_k"], 
        queries: Float[torch.Tensor, "batch_size ... seq_len_q d_k"], 
        values: Float[torch.Tensor, "batch_size ... seq_len_k d_v"],
        mask: Bool[torch.Tensor | None, "... seq_len_q seq_len_k"]=None,
        all_dim_masked: bool=False,
    ) -> Float[torch.Tensor, " ... seq_len_q d_v"]:
    d_k = keys.shape[-1]
    qk_transpose = einsum(queries, keys, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    qk_transpose = qk_transpose / d_k ** (0.5)
    if mask is not None:
        qk_transpose = qk_transpose.masked_fill(~mask, float("-inf")) 
    prob = softmax(qk_transpose, dim=-1, all_dim_masked=all_dim_masked)
    sdpa = einsum(prob, values, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return sdpa


class MHSA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_emb: nn.Module | None=None, device: torch.device | None=None, dtype: torch.dtype | None=None, context_length: int | None=None, use_qk_norm: bool=False,) -> None:
        super().__init__()
        self.num_heads = num_heads
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        total_dim = num_heads * (2 * d_k + d_v) # k, q, v head dim
        self.qkv_proj = Linear(d_model, total_dim, device=device, dtype=dtype)
        self.o_proj = Linear(int(num_heads * d_v), d_model, device=device, dtype=dtype)
        self.rope_emb = rope_emb
        self.context_length = context_length
        if self.context_length is None and self.rope_emb is not None:
            self.context_length = self.rope_emb.max_seq_len
        self.register_buffer("kv_cache", None, persistent=False)
        self.cache_len = 0
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(d_k, device=device, dtype=dtype)
            self.k_norm = RMSNorm(d_k, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_pos=None, mask: Bool[torch.Tensor | None, "seq_len seq_len"]=None, use_cache: bool=False) -> Float[torch.Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]

        # Add kv caching for inference
        use_cache = use_cache and not self.training
        if use_cache:
            qkv = self.qkv_proj(x)
            qkv = rearrange(qkv, "... seq_len (three h d_k) -> three ... h seq_len d_k", h=self.num_heads, three=3)
            q = qkv[0, ...]
            kv = qkv[1:, ...]

            assert self.context_length is not None, "context_length is needed to use the kv cache"
            assert seq_len <= self.context_length
            # allocate the cache once and reuse it
            if (self.kv_cache is None 
                or self.kv_cache.shape[:-2] != kv.shape[:-2] 
                or self.kv_cache.dtype != kv.dtype
                ):
                self.kv_cache = torch.empty((*kv.shape[:-2], self.context_length, kv.shape[-1]), dtype=kv.dtype, device=kv.device)
                self.cache_len = 0

            # drop the oldest tokens so the new ones fit in the cache
            curr_seq_len = min(self.cache_len, self.context_length - seq_len)
            if curr_seq_len < self.cache_len:
                start = self.cache_len - curr_seq_len
                self.kv_cache[..., :curr_seq_len, :] = self.kv_cache[..., start:self.cache_len, :].clone()
            self.kv_cache[..., curr_seq_len:curr_seq_len+seq_len, :] = kv
            self.cache_len = curr_seq_len + seq_len

            k, v = self.kv_cache[..., :self.cache_len, :]
            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if self.rope_emb is not None:
                # cannot exceed max len of rope embeddings
                assert curr_seq_len + seq_len <= self.rope_emb.max_seq_len
                # Using RoPE embeddings
                token_pos_k = torch.arange(curr_seq_len+seq_len, device=x.device)
                token_pos_q = torch.arange(start=curr_seq_len, end=curr_seq_len+seq_len, device=x.device)
                token_pos_k = rearrange(token_pos_k, "... seq_len -> ... 1 seq_len")
                token_pos_q = rearrange(token_pos_q, "... seq_len -> ... 1 seq_len")
                q = self.rope_emb(q, token_pos_q)
                k = self.rope_emb(k, token_pos_k)

        else:
            qkv = self.qkv_proj(x)
            q, k, v = rearrange(qkv, "... seq_len (three h d_k) -> three ... h seq_len d_k", h=self.num_heads, three=3)
            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if self.rope_emb is not None:
                # Using RoPE embeddings
                if token_pos is None:
                    token_pos = torch.arange(seq_len, device=x.device)
                token_pos= rearrange(token_pos, "... seq_len-> ... 1 seq_len")
                q = self.rope_emb(q, token_pos)
                k = self.rope_emb(k, token_pos)

        if mask is None and not use_cache:
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=0)
        elif use_cache:
            mask = torch.ones((seq_len, curr_seq_len+seq_len), dtype=torch.bool, device=x.device)
            if curr_seq_len == 0:
                mask = torch.tril(mask, diagonal=0)

        mhsa = scaled_dot_product_attention(k, q, v, mask)
        mhsa = rearrange(mhsa, "... h seq_len d_v -> ... seq_len (h d_v)")

        return self.o_proj(mhsa)

    def reset_cache(self) -> None:
        # keep the allocated memory, only forget the cached tokens
        self.cache_len = 0


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_emb: nn.Module | None, device: torch.device | None=None, dtype: torch.dtype | None=None, context_length: int | None=None, use_qk_norm: bool=False):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mhsa = MHSA(d_model, num_heads, rope_emb, device=device, dtype=dtype, context_length=context_length, use_qk_norm=use_qk_norm)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_pos=None, mask=None, use_cache: bool=False) -> Float[torch.Tensor, "... seq_len d_model"]:
        residual = x
        x = self.rms_norm1(x)
        x = self.mhsa(x, token_pos, mask, use_cache)
        x = x + residual
        residual = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, num_layers:int, d_model: int, num_heads: int, d_ff: int, context_length: int, theta: float | None=None, device: torch.device | None=None, dtype: torch.dtype | None=None, use_qk_norm: bool=False,):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.rope_emb = None
        if theta is not None:
            self.rope_emb = RotaryPositionalEmbedding(theta, d_model//num_heads, context_length, device)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, self.rope_emb, device, dtype, context_length, use_qk_norm) for _ in range(num_layers))
        self.rms_norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, vocab_size, device, dtype)

    def forward(self, input_tokens: Int[torch.Tensor, "... seq_len"], token_pos=None, mask=None, use_cache: bool=False,) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.emb(input_tokens)
        for layer in self.layers:
            x = layer(x, token_pos, mask, use_cache)
        x = self.rms_norm_final(x)
        x = self.out_proj(x) # output: logits
        return x

    @torch.no_grad()
    def reset_kv_cache(self):
        for layer in self.layers:
            layer.mhsa.reset_cache()


if __name__ == "__main__":
    import math
    bs, seq_len, d_model = 3, 7, 8
    vocab_size = 100
    device = "mps"
    x = torch.rand((bs, seq_len, d_model))
    out_dim = 10
    linear = Linear(d_model, out_dim, device)
    y = linear(x.to("mps"))
    assert list(y.shape) == [bs, seq_len, out_dim]
    print(y.device)

    embedding = Embedding(vocab_size, d_model)
    a = torch.randint(high=99, size=(bs, seq_len)).to(torch.long)
    b = embedding(a)

    rms_norm = RMSNorm(d_model)
    y = rms_norm(x)
    assert list(y.shape) == list(x.shape)

    d_ff = math.ceil(((8 / 3) * d_model) / 64) * 64
    swiglu = SwiGLU(d_model, d_ff).to(device)
    y = swiglu(x.to(device))
    assert list(y.shape) == list(x.shape)

    num_heads = 2
    theta = 10000
    d_k = d_model // num_heads
    context_len = 500
    rope_emb = RotaryPositionalEmbedding(theta, d_k, context_len, device)
    token_positions = torch.arange(seq_len, device=device).to(torch.long)
    # assert x.shape == y.shape
    
    import torch.nn.functional as F
    y = softmax(x, dim=2)
    y_ = F.softmax(x, dim=2)
    assert torch.allclose(y, y_)
    # print(y.shape)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask = mask[..., :-1]
    mask[..., -1] = False 
    y = scaled_dot_product_attention(x, x, x, mask)
    assert x.shape == y.shape

    num_heads = 2
    mhsa = MHSA(d_model, num_heads, device=device)
    y = mhsa(x.to("mps"))
    d_ff = (5 // 3) * d_model
    t_block = TransformerBlock(d_model, num_heads, d_ff, rope_emb=rope_emb, device=device)
    y = t_block(x.to(device))
    assert y.shape == x.shape

    a = torch.randint(high=99, size=(bs, seq_len)).to(torch.long)
    xformer = TransformerLM(vocab_size, 2, d_model, num_heads, d_ff, context_len, theta=theta, device=device)
    y = xformer(a)
    assert tuple(y.shape) == tuple((*a.shape, vocab_size))
