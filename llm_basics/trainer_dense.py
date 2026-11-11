import os
import math
import time
import argparse
from dataclasses import dataclass
from dataclasses import fields
import time
import json
import numpy as np
import torch
from typing import Any
from einops import rearrange, einsum, reduce
from llm_basics.layers import TransformerLM
from llm_basics.train_utils import get_batch
from llm_basics.train_utils import cross_entropy
from llm_basics.train_utils import gradient_clipping
from llm_basics.train_utils import lr_cosine_schedule
from llm_basics.train_utils import AdamW
from llm_basics.train_utils import load_checkpoint
from llm_basics.train_utils import save_checkpoint
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str = "default"
    vocab_size: int = 10000
    num_layers: int = 12
    d_model: int = 512
    num_heads: int = 4
    d_ff: int | None = None
    context_length: int = 512

    train_filepath: str = "data/TinyStoriesV2-GPT4-train.bin"
    validation_filepath: str = "data/TinyStoriesV2-GPT4-valid.bin"
    checkpoint_path: str = "checkpoints/llm_basics"
    batch_size: int = 4

    save_every: int = 100
    max_iter: int = 10000

    lr: float = 1e-5
    weight_decay: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999

    theta: int = 10000
    device: str | None = "mps"
    dtype: str | None = None

    min_lr: float = 1e-8
    warm_up_iter: int = 2000
    max_l2_norm: float = 2.0
    grad_accumulation: int = 1
    loss_scale_growth_interval: int = 2000

    do_validation: bool = True
    validation_steps: int | None = 20
    resume_from_checkpoint: str | None = None
    z_loss_weight: float | None = None # 1e-4 / 1e-5
    use_qk_norm: bool = False

    def update_from_dict(self, data: dict[str, Any]) -> None:
        valid_names = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in valid_names:
                setattr(self, key, value)

        # d_ff should be a multiple of 64
        d_ff = 8 / 3 * self.d_model
        self.d_ff = math.ceil(d_ff / 64) * 64

        if self.grad_accumulation is None:
            self.grad_accumulation = 1

def convert_dtype(dtype: str):
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "fp16":
            return torch.float16
        elif dtype in ["float", "float32"]:
            return torch.float32
        else: raise ValueError("Unknown datatype, select from bfloat16, fp16 or float32")


def train(config):
    # load dataset from disk
    train_dataset = np.memmap(config.train_filepath, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(config.validation_filepath, dtype=np.uint16, mode="r")
    mask = torch.ones((config.context_length, config.context_length), dtype=torch.bool, device=config.device)
    mask = torch.tril(mask, diagonal=0)

    dtype = convert_dtype(config.dtype)
    # fp16 keeps the weights in float32 and runs the forward pass in fp16 with autocast
    use_fp16 = dtype == torch.float16
    model_dtype = None if use_fp16 else dtype
    device_type = torch.device(config.device).type
    # model
    model = TransformerLM(config.vocab_size, config.num_layers, config.d_model, 
                          config.num_heads, config.d_ff, config.context_length, 
                          config.theta, config.device, model_dtype, use_qk_norm=config.use_qk_norm,
                          )
    logging.debug(model)
    total_params = sum(list(param.numel() for param in model.parameters()))
    print(f"Total number of parameters: {total_params / 1e6:.2f} M parameters")
    last_step = 0
    # optimizer
    opt = AdamW(model.parameters(), config.lr, config.weight_decay, (config.beta1, config.beta2))
    loss_scale = torch.tensor(65536.0, device=config.device) if use_fp16 else None
    inv_loss_scale = torch.empty((), device=config.device) if loss_scale is not None else None
    found_inf = torch.zeros((), device=config.device) if loss_scale is not None else None
    growth_tracker = 0
    if config.resume_from_checkpoint:
        logging.info(f"Resuming from saved checkpoint: {config.resume_from_checkpoint}")
        last_step, loss_scale_state = load_checkpoint(config.resume_from_checkpoint, model, opt, return_loss_scale=True)
        if loss_scale is not None and loss_scale_state is not None:
            loss_scale.fill_(loss_scale_state["scale"])
            growth_tracker = loss_scale_state["growth_tracker"]
    save_every = config.save_every

    os.makedirs(config.checkpoint_path, exist_ok=True)
    with open(os.path.join(config.checkpoint_path, f"{config.name}_training_logs.txt"), "a+") as file_writer:
        start_time = time.perf_counter()
        for step in range(last_step, config.max_iter):
            lr_step = lr_cosine_schedule(step, config.min_lr, config.lr, config.warm_up_iter, config.max_iter)
            print(f"{step},{"lr_step"},{lr_step}", file=file_writer)
            # update lr in opt
            for group in opt.param_groups:
                group["lr"] = lr_step

            train_losses = []
            z_losses = []
            for i in range(config.grad_accumulation):
                inputs, target = get_batch(train_dataset, config.batch_size, config.context_length, config.device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                    logits = model(inputs, mask=mask)
                ce = cross_entropy(logits, target, config.z_loss_weight is not None)
                # Scale loss
                if config.z_loss_weight:
                    total_train_loss = ce.loss + config.z_loss_weight * ce.z_loss
                else:
                    total_train_loss = ce.loss
                total_train_loss /= float(config.grad_accumulation)
                if loss_scale is not None:
                    (total_train_loss * loss_scale).backward()
                else:
                    total_train_loss.backward()
                train_losses.append(ce.loss.detach() / float(config.grad_accumulation))
                if config.z_loss_weight is not None:
                    z_losses.append(ce.z_loss.detach() / float(config.grad_accumulation))
            step_skipped = False
            if loss_scale is not None:
                found_inf.zero_()
                torch.reciprocal(loss_scale, out=inv_loss_scale)
                torch._amp_foreach_non_finite_check_and_unscale_(
                    [param.grad for param in model.parameters() if param.grad is not None],
                    found_inf,
                    inv_loss_scale,
                )
                step_skipped = bool(found_inf.item())

            if step_skipped:
                loss_scale.mul_(0.5).clamp_(min=1.0)
                growth_tracker = 0
                grad_norm = torch.full((), float("inf"), device=config.device)
            else:
                grad_norm = gradient_clipping(model.parameters(), config.max_l2_norm)
                opt.step()
                if loss_scale is not None:
                    growth_tracker += 1
                    if growth_tracker == config.loss_scale_growth_interval:
                        loss_scale.mul_(2.0)
                        growth_tracker = 0
            opt.zero_grad()
            print(f"{step},{"grad_norm"},{grad_norm.item()}", file=file_writer)
            train_loss = torch.stack(train_losses).sum(dim=0)
            train_loss = train_loss.item()
            logging.info(f"Train loss at step {step}: {train_loss:.2f}")
            print(f"{step},{"train_loss"},{train_loss}", file=file_writer)
            if z_losses:
                z_loss = torch.stack(z_losses).sum(dim=0)
                print(f"{step},{"z_loss"},{z_loss.item()}", file=file_writer)
            
            model.eval()
            val_loss = 0.0
            if config.do_validation and (step + 1) % save_every == 0:
                val_loss = torch.tensor([0.0], device=config.device)
                with torch.no_grad():
                    for _ in range(config.validation_steps):
                        val_inputs, val_target = get_batch(valid_dataset, config.batch_size, config.context_length, config.device)
                        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                            val_logits = model(val_inputs, mask=mask)
                        _val_loss = cross_entropy(val_logits, val_target)
                        val_loss += _val_loss.loss
                    val_loss = val_loss.item()
                    val_loss /= float(config.validation_steps)
                    logging.info(f"validation loss at step {step}: {val_loss:.2f}")
                    print(f"{step},{"validation_loss"},{val_loss}", file=file_writer)
            model.train()
            if (step + 1) % save_every == 0:
                output_file = f"{config.name}_model_ckpt_{time.time()}_step={step + 1}_val_loss={val_loss}.pt"
                full_save_path = os.path.join(config.checkpoint_path, output_file)
                loss_scale_state = None
                if loss_scale is not None:
                    loss_scale_state = {"scale": loss_scale.item(), "growth_tracker": growth_tracker}
                save_checkpoint(model, opt, step + 1, full_save_path, loss_scale_state)
                logging.info(f"Saving checkpoint at iteration {step} in {output_file}...")

            print(f"{step},{"elapsed_time"},{time.perf_counter() - start_time}", file=file_writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decoder-only LLM on a dataset")
    parser.add_argument("config", type=str, help="Path to the training config")

    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as file_read:
        config_dict = json.load(file_read)
    config = ModelConfig()
    config.update_from_dict(config_dict)
    print(config)
    train(config)
