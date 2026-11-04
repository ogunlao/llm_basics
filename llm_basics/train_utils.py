from collections.abc import Callable
import os
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce
from jaxtyping import Float
from jaxtyping import Int
from typing import Iterable
import typing 



def cross_entropy(logits: Float[torch.Tensor, "... vocab_size"], target: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, ""]:
    logits = logits.to(torch.float32)
    max_logits = reduce(logits, "... vocab_size -> ... 1", "max")
    logits = logits - max_logits
    logit_exp = torch.exp(logits)
    logit_sum_exp = reduce(logit_exp, "... vocab_size -> ... 1", "sum")
    target = rearrange(target, "... -> ... 1")
    target_logits = torch.gather(logits, dim=-1, index=target)
    loss = - target_logits + torch.log(logit_sum_exp)
    loss = reduce(loss, "... 1 -> 1", "mean")
    return loss


def perplexity(losses: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, ""]:
    return torch.exp(reduce(losses, "... -> 1", "mean"))

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float=0.0, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}. betas should be between 0.0 and 1.0")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}. eps should be positive")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Float[torch.Tensor ,""] | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]  # Get state associated with p.

                # lazy initialization of moments
                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(p)
                    state["second_moment"] = torch.zeros_like(p)

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                state["step"] += 1

                step = state["step"]
                lr_t = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)

                # Apply weight decay
                p -= lr * weight_decay * p

                first_moment *= beta1
                first_moment += (1 - beta1) * grad

                second_moment *= beta2
                second_moment += (1 - beta2) * grad**2

                p -= lr_t * (first_moment / (second_moment**(0.5) + eps))
        return loss

def lr_cosine_schedule(step: int, min_lr: float, max_lr: float, total_warm_up_iter: int, final_iter: int) -> float:
    if step < total_warm_up_iter:
        lr_t = step *  max_lr / total_warm_up_iter
    elif total_warm_up_iter <= step <= final_iter:
        lr_t = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos((step - total_warm_up_iter) * math.pi / (final_iter - total_warm_up_iter)))
    else:
        lr_t = min_lr
    return lr_t

def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> Float[torch.Tensor, ""]:
    parameters = list(parameters)
    device = parameters[0].device if parameters else None
    max_l2_norm = torch.tensor(max_l2_norm, device=device)
    frob_norm = torch.tensor([0.0], device=device)
    for param in parameters:
        # compute norm of gradient
        if param.grad is not None:
            frob_norm += torch.sum(param.grad ** 2)

    frob_norm = torch.sqrt(frob_norm)
    _frob_norm = torch.max(frob_norm, max_l2_norm)
    scaling = (max_l2_norm / (_frob_norm + eps))
    for param in parameters:
        if param.grad is not None:
            param.grad *= scaling
    return frob_norm

def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    x_list = [dataset[i : i + context_length] for i in start_indices]
    y_list = [dataset[i + 1 : i + 1 + context_length] for i in start_indices]

    x = rearrange(x_list, "batch context_len -> batch context_len")
    y = rearrange(y_list, "batch context_len -> batch context_len")

    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)

    return x, y


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    # Unwrap torch.compile so saved keys don't carry the "_orig_mod." prefix
    model = getattr(model, "_orig_mod", model)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer | None = None,
                    map_location: str | None = None,
                    ):
    checkpoints = torch.load(src, map_location=map_location)
    model.load_state_dict(checkpoints["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoints["optimizer"])

    return checkpoints["step"]


if __name__ == "__main__":
    batch_size = 5
    seq_len = 8
    vocab_size = 30
    logits = torch.randn((batch_size, seq_len, vocab_size))

    # Target token IDs at each position
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    loss = cross_entropy(logits, targets)
    print(loss)

    loss_ = torch.nn.functional.cross_entropy(logits.transpose(-2, -1), targets)
    print(loss_)
    assert torch.allclose(loss, loss_)

    x = torch.randn(seq_len)
    ppl = perplexity(x)
    print(ppl)

    y = torch.randn(seq_len, requires_grad=True)
    loss = (y**2).sum()
    opt = AdamW([y], lr=0.001)
    opt.zero_grad()
    loss.backward()
    opt.step()

    gradient_clipping([y.to("mps")], 1.0)