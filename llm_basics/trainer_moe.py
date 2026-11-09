import os
import math
import time
import argparse
import time
import json
import numpy as np
import torch
from dataclasses import dataclass
from dataclasses import fields
import logging
from typing import Any
from einops import rearrange, einsum, reduce
from llm_basics.layers import Linear
from llm_basics.layers import SwitchTransformerLM
from llm_basics.train_utils import get_batch
from llm_basics.train_utils import cross_entropy
from llm_basics.train_utils import gradient_clipping
from llm_basics.train_utils import lr_cosine_schedule
from llm_basics.train_utils import AdamW
from llm_basics.train_utils import load_checkpoint
from llm_basics.train_utils import save_checkpoint


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@torch.no_grad()
def init_layer(m):
    if isinstance(m, Linear):
        rho = 0.1 / m.input_features
        rho = rho ** (0.5)
        torch.nn.init.trunc_normal_(m.weights, mean=0.0, std=rho, a=-3*rho, b=3*rho)

@dataclass
class ModelConfig:
    name: str = "default"
    num_experts: int = 2
    capacity_factor: float | None = None
    aux_loss_weight: float = 1e-2
    jitter_noise: float = 1e-2

    vocab_size: int = 10000
    num_layers: int = 12
    d_model: int = 512
    num_heads: int = 4
    d_ff: int | None = None
    context_length: int = 512

    train_filepath: str = "data/TinyStoriesV2-GPT4-train.bin"
    validation_filepath: str = "data/TinyStoriesV2-GPT4-valid.bin"
    checkpoint_path: str = "checkpoints/"
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

    do_validation: bool = True
    validation_steps: int | None = 20
    resume_from_checkpoint: str | None = None
    z_loss_weight: float | None = None # 1e-4 / 1e-5

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
    # model
    model = SwitchTransformerLM(config.num_experts, config.capacity_factor, 
                                config.vocab_size, config.num_layers, config.d_model, 
                                config.num_heads, config.d_ff, config.context_length, 
                                config.theta, jitter_noise=config.jitter_noise,
                                device=config.device, dtype=dtype,
                          )
    model.apply(init_layer)
    logging.debug(model)
    # Automatically find more optimized kernels
    # model = torch.compile(model, backend="aot_eager")
    total_params = sum(list(param.numel() for param in model.parameters()))
    print(f"Total number of parameters: {total_params / 1e6:.2f} M parameters")
    last_step = 0
    # optimizer
    opt = AdamW(model.parameters(), config.lr, config.weight_decay, (config.beta1, config.beta2))
    if config.resume_from_checkpoint:
        logging.info(f"Resuming from saved checkpoint: {config.resume_from_checkpoint}")
        last_step = load_checkpoint(config.resume_from_checkpoint, model, opt)
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
            train_aux_losses = []
            z_losses = []
            for i in range(config.grad_accumulation):
                inputs, target = get_batch(train_dataset, config.batch_size, config.context_length, config.device)
                logits, train_aux_loss = model(inputs, mask=mask)
                ce = cross_entropy(logits, target, config.z_loss_weight is not None)

                train_loss = ce.loss
                train_loss /= float(config.grad_accumulation)
                train_aux_loss /= float(config.grad_accumulation)
                total_train_loss = train_loss + config.aux_loss_weight * train_aux_loss
                if config.z_loss_weight is not None:
                    z_loss = ce.z_loss
                    z_loss /= float(config.grad_accumulation)
                    total_train_loss += config.z_loss_weight * z_loss
                    z_losses.append(z_loss.detach())
                
                # Scale loss
                total_train_loss.backward()
                train_losses.append(train_loss.detach())
                train_aux_losses.append(train_aux_loss.detach())

            grad_norm = gradient_clipping(model.parameters(), config.max_l2_norm)
            opt.step()
            opt.zero_grad()
            print(f"{step},{"grad_norm"},{grad_norm.item()}", file=file_writer)
            train_loss = torch.stack(train_losses).sum(dim=0)
            train_aux_loss = torch.stack(train_aux_losses).sum(dim=0)
            train_loss = train_loss.item()
            train_aux_loss = train_aux_loss.item()
            logging.info(f"Train loss at step {step}: {train_loss:.2f}")
            print(f"{step},{"train_loss"},{train_loss}", file=file_writer)
            print(f"{step},{"train_aux_loss"},{train_aux_loss}", file=file_writer)
            if z_losses:
                z_loss = torch.stack(z_losses).sum(dim=0)
                print(f"{step},{"z_loss"},{z_loss.item()}", file=file_writer)

            val_loss = 0.0
            if config.do_validation and (step + 1) % save_every == 0:
                val_loss = torch.tensor([0.0], device=config.device)
                val_aux_loss = torch.tensor([0.0], device=config.device)
                # Eval mode disables expert capacity and router jitter noise
                model.eval()
                with torch.no_grad():
                    for _ in range(config.validation_steps):
                        val_inputs, val_target = get_batch(valid_dataset, config.batch_size, config.context_length, config.device)
                        val_logits, validation_aux_loss = model(val_inputs, mask=mask)
                        _val_loss = cross_entropy(val_logits, val_target)
                        val_loss += _val_loss.loss
                        val_aux_loss += validation_aux_loss

                    val_loss = val_loss.item()
                    val_loss /= float(config.validation_steps)
                    val_aux_loss = val_aux_loss.item()
                    val_aux_loss /= float(config.validation_steps)
                    logging.info(f"validation loss at step {step}: {val_loss:.2f}")
                    print(f"{step},{"validation_loss"},{val_loss}", file=file_writer)
                    print(f"{step},{"validation_aux_loss"},{val_aux_loss}", file=file_writer)
                model.train()

            if (step + 1) % save_every == 0:
                output_file = f"{config.name}_model_ckpt_{time.time()}_step={step + 1}_val_loss={val_loss}.pt"
                full_save_path = os.path.join(config.checkpoint_path, output_file)
                save_checkpoint(model, opt, step + 1, full_save_path)
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