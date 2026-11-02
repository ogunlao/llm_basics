import json
import os
import time
from llm_basics.train_bpe import train_bpe

if __name__ == "__main__":
    input_path = "data/owt_train.txt"
    vocab_size = 32_000
    num_processes = os.cpu_count() or 1
    chunk_size_mb = 4 # set with profiling
    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, num_processes=num_processes, chunk_size_mb=chunk_size_mb)
    duration = time.perf_counter() - start
    print(f"Training took {duration:.2f} s")

    with open("data/owt_vocab.json", "w") as f:
        serialized_vocab = {k: v.decode("latin-1") for k, v in vocab.items()}
        json.dump(serialized_vocab, f, indent=2, ensure_ascii=False)

    with open("data/owt_merges.json", "w", encoding="utf-8") as f:
        serialised_merges = [(tk1.decode("latin-1"), tk2.decode("latin-1")) for tk1, tk2 in merges]
        json.dump(serialised_merges, f, ensure_ascii=False)

    