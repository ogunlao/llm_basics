import numpy as np
from llm_basics.tokenizer.load_bpe import Tokenizer


def encode_data(
    data_filepath: str, 
    output_filepath: str, 
    vocab_filepath: str, 
    merges_filepath: str, 
    special_tokens: list[str] = ["<|endoftext|>"],
    chunk_size: int = 1_000_000,
    dtype: np.dtype = np.uint16
):
    bpe_tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    buffer = []
    with open(data_filepath, "r", encoding="utf-8") as f_in, open(output_filepath, "wb") as f_out:
        for token in bpe_tokenizer.encode_iterable(f_in):
            # If token is a tensor, convert to int; if it yields batches, extend instead
            # our iterator normally yield list
            if hasattr(token, "item"):
                buffer.append(token.item())
            elif hasattr(token, "tolist"):
                buffer.extend(token.tolist())
            else:
                buffer.append(int(token))

            # Flush to disk when buffer reaches chunk_size
            if len(buffer) >= chunk_size:
                np.array(buffer, dtype=dtype).tofile(f_out)
                buffer.clear()

        # Flush any remaining tokens
        if buffer:
            np.array(buffer, dtype=dtype).tofile(f_out)

if __name__ == "__main__":
    for data in ["valid", "train"]:
        encode_data(
            data_filepath=f"data/TinyStoriesV2-GPT4-{data}.txt", 
            output_filepath=f"data/TinyStoriesV2-GPT4-{data}.bin", 
            vocab_filepath=f"data/tinystories_vocab.json", 
            merges_filepath=f"data/tinystories_merges.json",
        )

    # for data in ["valid", "train"]:
    #     encode_data(
    #         data_filepath=f"data/owt_{data}.txt", 
    #         output_filepath=f"data/owt_{data}.bin", 
    #         vocab_filepath=f"data/owt_vocab.json", 
    #         merges_filepath=f"data/owt_merges.json",
    #     )