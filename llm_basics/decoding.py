import argparse
import json
from dataclasses import dataclass
import logging
import torch
from einops import rearrange
from llm_basics.trainer_dense import ModelConfig
from llm_basics.trainer_moe import ModelConfig as MoEModelConfig
from llm_basics.layers import TransformerLM
from llm_basics.layers import SwitchTransformerLM
from llm_basics.tokenizer.load_bpe import Tokenizer
from llm_basics.layers import softmax
from llm_basics.train_utils import load_checkpoint
from jaxtyping import Float
from jaxtyping import Int

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class DecodingConfig:
    vocab_filepath: str
    merges_filepath: str
    checkpoint_path: str
    max_tokens: int = 100
    softmax_temperature: float = 1.0
    top_p: float | None = 1.0
    use_cache: bool = False
    

def load_tokenizer(decoding_config):
    bpe_tokenizer = Tokenizer.from_files(decoding_config.vocab_filepath, decoding_config.merges_filepath, ["<|endoftext|>"])
    return bpe_tokenizer


def load_model(config: ModelConfig | MoEModelConfig, checkpoint_path: str) -> torch.nn.Module:
    if isinstance(config, MoEModelConfig):
        model = SwitchTransformerLM(
            config.num_experts, config.capacity_factor,
            config.vocab_size, config.num_layers, config.d_model, config.num_heads,
            config.d_ff, config.context_length, config.theta,
            jitter_noise=config.jitter_noise, device=config.device, use_qk_norm=config.use_qk_norm,
        )
    else:
        model = TransformerLM(
            config.vocab_size, config.num_layers, config.d_model, config.num_heads,
            config.d_ff, config.context_length, config.theta, config.device, use_qk_norm=config.use_qk_norm,
        )
    _ = load_checkpoint(checkpoint_path, model, None, config.device)
    model.eval()
    return model


def get_token_pred_from_logit(logits: Float[torch.Tensor, "bs seq_len dim"], softmax_temperature: float, top_p: float) -> Int[torch.Tensor, "bs 1"]:
    logits = logits[:, -1, :]
    if softmax_temperature == 0:
        # pick the most likely token
        index = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits /= softmax_temperature

        probs = softmax(logits, dim=-1)

        # top p sampling
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum = torch.cumsum(sorted_probs, dim=-1)
        masks = cum_sum >= top_p
        masks[..., 1:] = masks[..., :-1].clone()
        masks[..., 0] = 0

        sorted_probs[masks] = 0.0
        filtered_probs = torch.zeros_like(probs).scatter_(
            dim=-1, index=sorted_indices, src=sorted_probs
        )
        filtered_probs = rearrange(filtered_probs, "1 probs -> probs")
        index = filtered_probs.multinomial(num_samples=1, replacement=True)
        index = torch.unsqueeze(index, 0)
    return index


def decode(prefix: str, config: ModelConfig, dec_config: DecodingConfig) -> str:
    logging.debug("Loading model ...")
    model = load_model(config, dec_config.checkpoint_path)
    softmax_temperature = dec_config.softmax_temperature
    max_tokens = dec_config.max_tokens
    top_p = 1.0 if dec_config.top_p is None else dec_config.top_p

    tokenizer = load_tokenizer(dec_config)
    end_of_text_index = tokenizer.encode("<|endoftext|>")[0]
    end_of_text_index = torch.tensor([end_of_text_index], dtype=torch.long, device=config.device)

    if prefix:
        tokens = tokenizer.encode(prefix)
    else:
        tokens = [end_of_text_index]
    tokens = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0) # bs, seq

    total_gen_tokens = 0
    if dec_config.use_cache:
        model.reset_kv_cache()

    warmed_up_cache = False
    while True:
        with torch.no_grad():
            _tokens = tokens[:, -config.context_length:]
            
            if dec_config.use_cache and not warmed_up_cache:
                logits = model(_tokens, use_cache=dec_config.use_cache)
                if isinstance(logits, tuple):
                    logits, _ = logits
                next_index = get_token_pred_from_logit(logits, softmax_temperature, top_p)
                if next_index[:, 0] == end_of_text_index:
                    logging.debug("Finished inference as model predicted end of text.")
                    break
                tokens = torch.cat([tokens, next_index], dim=1)
                warmed_up_cache = True
                total_gen_tokens += 1
                if total_gen_tokens >= max_tokens:
                    break

            if dec_config.use_cache:
                # feed only the last token
                logits = model(next_index, use_cache=dec_config.use_cache)
            else:
                logits = model(_tokens, use_cache=dec_config.use_cache)
            if isinstance(logits, tuple):
                logits, _ = logits

            next_index = get_token_pred_from_logit(logits, softmax_temperature, top_p)
            if next_index[:, 0] == end_of_text_index:
                logging.debug("Finished inference as model predicted end of text.")
                break
            tokens = torch.cat([tokens, next_index], dim=1)

        total_gen_tokens += 1

        if total_gen_tokens >= max_tokens:
            break

    if tokens[:, 0] == end_of_text_index:
        tokens = tokens[:, 1:]

    text = tokenizer.decode(tokens[0].cpu().tolist())
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode from a decoder-only LLM")
    parser.add_argument("prefix", default="", type=str, nargs="?", help="Prefix to use to decode from the LLM")
    parser.add_argument("config", type=str, help="Path to the model config")
    parser.add_argument("decoder_config", type=str, help="Path to the decoder config")

    args = parser.parse_args()
    model_config_path = args.config
    with open(model_config_path) as file_read:
        config_dict = json.load(file_read)
    model_config = MoEModelConfig() if "num_experts" in config_dict else ModelConfig()
    model_config.update_from_dict(config_dict)
    print(model_config)

    dec_config_path = args.decoder_config
    with open(dec_config_path) as file_read:
        decoding_config_dict = json.load(file_read)
    decoding_config = DecodingConfig(**decoding_config_dict)
    text = decode(args.prefix, model_config, decoding_config)
    print(text)
