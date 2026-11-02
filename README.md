# From Scratch Implementation of Decoder-Style LLM like GPT-2

It includes a byte-level BPE tokenizer, a Transformer LM (RMSNorm, SwiGLU, RoPE, causal multi-head self-attention), an AdamW optimizer, a cosine learning-rate schedule, gradient clipping, a training loop with checkpointing, and top-p decoding.

## Project structure

```
llm_basics/
├── train_bpe.py            # BPE tokenizer training (fast heap-based + slow reference version)
├── load_bpe.py             # Tokenizer: encode / decode using a trained vocab + merges
├── text_loading.py         # Parallel, chunked pre-tokenization of large text files
├── layers.py               # Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, TransformerLM
├── training.py             # Cross-entropy, AdamW, LR schedule, gradient clipping, batching, checkpoints
├── trainer.py              # Training loop (entry point)
├── decoding.py             # Text generation from a trained checkpoint (entry point)
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

### Train a small model on the TinyStories dataset
```sh
uv run llm_basics/trainer.py "llm_basics/config/tiny_stories_config.json"
```
Checkpoints and training logs are written to `checkpoints/tiny_stories/`.

Notes on `llm_basics/config/tiny_stories_config.json`:
- `device` is set to `"mps"` (Apple Silicon). Change it to `"cuda"` or `"cpu"` for other hardware.
- To resume training, set `resume_from_checkpoint` to the path of a saved `.pt` checkpoint.

### Run inference on the trained model
First set `checkpoint_path` in `llm_basics/config/tiny_stories_decoding_config.json` to a checkpoint from `checkpoints/tiny_stories/`, then run
```sh
uv run llm_basics/decoding.py "<PREFIX_TEXT>" "llm_basics/config/tiny_stories_config.json" "llm_basics/config/tiny_stories_decoding_config.json"
```

The example can be adapted to train on OpenWebText or other datasets (see `llm_basics/tokenizer_training/train_bpe_expts_owt.py`).

The implementation follows closely [Stanford CS336](https://stanford-cs336.github.io/) (Assignment 1).