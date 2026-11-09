import os
import collections
import heapq
from llm_basics.tokenizer.text_loading import get_streaming_pretokens
from llm_basics.tokenizer.text_loading import get_merges


def compute_pretoken_bytes(pretokens) -> dict[tuple[bytes, ...], int]:
    pretok_rep = {}
    for pt, count in pretokens.items():
        pt_tuple = tuple(bytes([c]) for c in pt.encode("utf-8"))
        pretok_rep[pt_tuple] = count
    return pretok_rep


class Merge:
    __slots__ = ("merge", "count")
    def __init__(self, merge: tuple[bytes, bytes], count: int):
        self.merge = merge
        self.count = count

    def __lt__(self, other):
        return (self.count, self.merge) > (other.count, other.merge)
    
def train_bpe(input_path: str | os.PathLike, 
              vocab_size: int, 
              special_tokens: list[str] = ["<|endoftext|>"],
              num_processes: int=4, 
              chunk_size_mb: int=32, 
    ):
    # Initialise default vocabulary of 256 code points
    vocab: dict[int, bytes] = {}
    merge_list = []

    # Add special tokens
    if special_tokens:
        for idx, special_tok in enumerate(special_tokens):
            vocab[idx] = special_tok.encode("utf-8")
    last_index = 0
    if vocab:
        last_index = max(vocab.keys()) + 1

    for idx in range(256):
        vocab[last_index + idx] = bytes([idx])
    next_idx = len(vocab)

    rem_merges = vocab_size - next_idx
    if rem_merges <= 0:
        return vocab, []
    
    pretokens = collections.Counter() # {" the": 5, "'ve": 10, ...}
    # Steam pretokens to reduce memory
    for batch_pretokens in get_streaming_pretokens(input_path, special_tokens, num_processes, chunk_size_mb):
        pretokens.update(batch_pretokens)

    pretoken_count: dict[tuple[bytes, ...], int] = compute_pretoken_bytes(pretokens) # {(b" ", b"t", b"h", b"e"): 5, (b"'", b"v", b"e"): 10, ...}
    
    merges, merges_tracker = get_merges(pretoken_count)
    merges_heap = [Merge(merge, count) for merge, count in merges.items()]
    # Helps to get max in constant time
    heapq.heapify(merges_heap)

    def get_max_byte(merges) -> tuple[bytes, bytes] | None:
        while merges_heap:
            max_merge: Merge = heapq.heappop(merges_heap)
            max_byte_pair, count =  max_merge.merge, max_merge.count
            # The heap is not a one-to-one represntation of "merges"
            if max_byte_pair not in merges or merges[max_byte_pair] != count:
                continue
            else:
                return max_byte_pair
        return None

    def update_byte_tuple(byte_tuple, max_byte_pair):
        # update affected byte string with merges
        new_byte_tuple = []
        byte_str: bytes = b"".join(max_byte_pair)
        i = 0
        while i < len(byte_tuple):
            if i + 1 < len(byte_tuple) and byte_tuple[i] == max_byte_pair[0] and byte_tuple[i+1] == max_byte_pair[1]:
                new_byte_tuple.append(byte_str)
                i += 1
            else:
                new_byte_tuple.append(byte_tuple[i])
            i += 1
        return tuple(new_byte_tuple)

    for i in range(rem_merges):
        if not merges:
            break
        max_byte_pair: tuple[bytes, bytes] | None = get_max_byte(merges)

        if max_byte_pair is None:
            break

        byte_tuples_to_update = list(merges_tracker[max_byte_pair])
        # update merges
        byte_tuple_pair = [] # (old, new)
        byte_pair_str: bytes = b"".join(max_byte_pair)

        for byte_tuple in byte_tuples_to_update:
            new_byte_tuple = update_byte_tuple(byte_tuple, max_byte_pair)
            byte_tuple_pair.append((byte_tuple, new_byte_tuple))
            # The new tuple takes the count of the old as usual
            pretoken_count[new_byte_tuple] = pretoken_count[byte_tuple]
            del pretoken_count[byte_tuple]
        
        merges_to_update = set()
        # Updating a specific token, needs to update surrounding 
        # affected tokens by that merge
        for curr_byte_tuple, new_byte_tuple in byte_tuple_pair:
            for tk1, tk2 in zip(new_byte_tuple, new_byte_tuple[1:]):
                if tk1 == tk2 == byte_pair_str:
                    # Both are the new byte pair
                    merges[(tk1, tk2)] += pretoken_count[new_byte_tuple]
                    merges[(max_byte_pair[1], max_byte_pair[0])] -= pretoken_count[new_byte_tuple]

                    merges_to_update.add((tk1, tk2))
                    merges_to_update.add((max_byte_pair[1], max_byte_pair[0]))

                    # Clean-up
                    if merges[(max_byte_pair[1], max_byte_pair[0])] == 0:
                        del merges[(max_byte_pair[1], max_byte_pair[0])]
                    if (max_byte_pair[1], max_byte_pair[0]) in merges_tracker:
                        merges_tracker[(max_byte_pair[1], max_byte_pair[0])].discard(curr_byte_tuple)
                        if len(merges_tracker[(max_byte_pair[1], max_byte_pair[0])]) == 0:
                            del merges_tracker[(max_byte_pair[1], max_byte_pair[0])]
                elif tk2 == byte_pair_str:
                    # Byte pair fall on the second token
                    merges[(tk1, tk2)] += pretoken_count[new_byte_tuple]
                    merges[(tk1, max_byte_pair[0])] -= pretoken_count[new_byte_tuple]

                    merges_to_update.add((tk1, tk2))
                    merges_to_update.add((tk1, max_byte_pair[0]))

                    if merges[(tk1, max_byte_pair[0])] == 0:
                        del merges[(tk1, max_byte_pair[0])]
                    if (tk1, max_byte_pair[0]) in merges_tracker:
                        merges_tracker[(tk1, max_byte_pair[0])].discard(curr_byte_tuple)
                        if len(merges_tracker[(tk1, max_byte_pair[0])]) == 0:
                            del merges_tracker[(tk1, max_byte_pair[0])]
                elif tk1 == byte_pair_str:
                    merges[(tk1, tk2)] += pretoken_count[new_byte_tuple]
                    merges[(max_byte_pair[1], tk2)] -= pretoken_count[new_byte_tuple]

                    merges_to_update.add((tk1, tk2))
                    merges_to_update.add((max_byte_pair[1], tk2))

                    if merges[(max_byte_pair[1], tk2)] == 0:
                        del merges[(max_byte_pair[1], tk2)]
                    if (max_byte_pair[1], tk2) in merges_tracker:
                        merges_tracker[(max_byte_pair[1], tk2)].discard(curr_byte_tuple)
                        if len(merges_tracker[(max_byte_pair[1], tk2)]) == 0:
                            del merges_tracker[(max_byte_pair[1], tk2)]
                # normal bytes without merges
                if curr_byte_tuple in merges_tracker[(tk1, tk2)]:
                    merges_tracker[(tk1, tk2)].remove(curr_byte_tuple) # expired tuple
                merges_tracker[(tk1, tk2)].add(new_byte_tuple) # updated byte tuple

        del merges[max_byte_pair] # no longer a valid pair
        if max_byte_pair in merges_tracker:
            del merges_tracker[max_byte_pair]
        # update heap
        for byte_tuple in merges_to_update:
            if byte_tuple in merges and merges[byte_tuple] > 0:
                heapq.heappush(merges_heap, Merge(byte_tuple, merges[byte_tuple]))

        merge_list.append(max_byte_pair)

    for i in range(len(merge_list)):
        vocab[next_idx + i] = b"".join(merge_list[i])
    return vocab, merge_list


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1_000
    special_tokens = ["<|endoftext|>"]
    num_processes = os.cpu_count() or 1

    # Check which memory chunk size is suitable for the current dataset
    profile_it = False
    if profile_it:
        import time
        target_sizes = [4, 16, 32, 64]
        file_path = "data/TinyStoriesV2-GPT4-valid.txt"
        for size_mb in target_sizes:
            start_time = time.perf_counter()
            train_bpe(path, vocab_size, num_processes=num_processes, chunk_size_mb=size_mb)
            elapsed = time.perf_counter() - start_time
            print(f"Target Chunk Size: {size_mb} MB | Chunks: | Time: {elapsed:.2f}s")

    vocab, merge_list = train_bpe(path, vocab_size, num_processes=num_processes, chunk_size_mb=4)



