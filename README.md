# From Scratch Implementation of Decoder-Style LLM. 

It includes a byte-level BPE tokenizer, a Transformer LM (RMSNorm, SwiGLU, RoPE, causal multi-head self-attention), Switch Transformer MOE LLM, an AdamW optimizer, a cosine learning-rate schedule, gradient clipping, a training loop with checkpointing, greedy/top-p decoding with KV caching, and an option for QK normalisation.

## Project structure

```
llm_basics/
├── layers/
│   ├── layers.py           # Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention (with KV cache), TransformerLM
│   └── moe_layers.py       # Switch Transformer MoE block and SwitchTransformerLM
├── tokenizer/
│   ├── load_bpe.py         # Tokenizer: encode / decode using a trained vocab + merges
│   └── text_loading.py     # Parallel, chunked pre-tokenization of large text files
├── train_bpe.py            # BPE tokenizer training (heap-based)
├── train_utils.py          # Cross-entropy, AdamW, LR schedule, gradient clipping, batching, checkpoints
├── trainer_dense.py        # Training loop for the dense model (entry point)
├── trainer_moe.py          # Training loop for the MoE model (entry point)
├── decoding.py             # Text generation from a trained dense or MoE checkpoint (entry point)
├── config/                 # Model/training and decoding configs
└── tokenizer_training/     # Scripts to train the tokenizer and pre-encode datasets
```

## Setup

### Environment
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.

Then install the dependencies:
```sh
uv sync
```

All commands below are run from the repository root.

### Download data
Download the TinyStories data or a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Train the BPE tokenizer on TinyStories
This trains a 10K vocabulary and writes `data/tinystories_vocab.json` and `data/tinystories_merges.json`.
```sh
uv run llm_basics/tokenizer_training/train_bpe_tinystories.py
```
Then, pre-encode the train/validation data into `data/TinyStoriesV2-GPT4-{train,valid}.bin` for model training
```sh
uv run llm_basics/tokenizer_training/encode_data.py
```

### Train a small dense model on the TinyStories dataset
```sh
uv run llm_basics/trainer_dense.py "llm_basics/config/tiny_stories_config.json"
```
Checkpoints and training logs are written to `checkpoints/tiny_stories/`.

### Train a small MOE-style model on the TinyStories dataset
```sh
uv run llm_basics/trainer_moe.py "llm_basics/config/tiny_stories_moe_config.json"
```
Checkpoints and training logs are written to `checkpoints/tiny_stories_moe/`.

Notes on the training configs (`tiny_stories_config.json`, `tiny_stories_moe_config.json`):
- `device` is set to `"mps"` (Apple Silicon). Change it to `"cuda"` or `"cpu"` for other hardware.
- To resume training, set `resume_from_checkpoint` to the path of a saved `.pt` checkpoint.
- For the MOE model, set `capacity_factor` to `null` to remove the expert capacity limit. The capacity limit and the router `jitter_noise` are only used in training, they are disabled during validation and decoding.

### Run inference on the trained model
First set `checkpoint_path` in `llm_basics/config/tiny_stories_decoding_config.json` to a checkpoint from `checkpoints/tiny_stories/`, then run
```sh
uv run llm_basics/decoding.py "<PREFIX_TEXT>" "llm_basics/config/tiny_stories_config.json" "llm_basics/config/tiny_stories_decoding_config.json"
```

For the MoE model, set `checkpoint_path` to a checkpoint from `checkpoints/tiny_stories_moe/` and pass the MoE config instead. The model type is picked from the config (a config with `num_experts` loads the MoE model)
```sh
uv run llm_basics/decoding.py "<PREFIX_TEXT>" "llm_basics/config/tiny_stories_moe_config.json" "llm_basics/config/tiny_stories_decoding_config.json"
```

Notes on the decoding config (`tiny_stories_decoding_config.json`):
- `softmax_temperature` set to `0.0` does greedy decoding. Set it above `0.0` to sample with `top_p` (nucleus sampling).
- `max_tokens` is the maximum number of new tokens to generate.
- `use_cache` turns on KV caching for both the dense and MOE models.
- Greedy decoding gives the same output with or without the cache until the sequence reaches `context_length`. After that, the outputs can differ.

The example can be adapted to train on OpenWebText or other datasets (see `llm_basics/tokenizer_training/train_bpe_expts_owt.py`).

The dense model is a LLaMA-style decoder (pre-norm RMSNorm, SwiGLU, RoPE) and the MOE implemented is the SwitchLayer in Switch Transformers. The project started from [Stanford CS336](https://stanford-cs336.github.io/) Assignment 1 and has since been extended with MOE and KV caching.