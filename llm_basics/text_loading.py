import os
import functools
import regex as re
import collections
from typing import BinaryIO
import multiprocessing as mp
from typing import Iterator

def get_merges(pretokens: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple, set]]:
    merges = collections.defaultdict(int)
    merges_tracker = collections.defaultdict(set)
    for byte_tuple, count in pretokens.items():
        if len(byte_tuple) < 1:
            continue
        for tk1, tk2 in zip(byte_tuple, byte_tuple[1:]):
            merges[(tk1, tk2)] += count
            # Track every byte array (word) that uses that byte pair
            merges_tracker[(tk1, tk2)].add(byte_tuple)
    return merges, merges_tracker

def replace_pretok(curr_pretok, byte_tuple) -> dict[tuple[bytes, ...], int]:
    new_pretok = collections.defaultdict(int)
    for k, v in curr_pretok.items():
        new_k = []
        i = 0
        while i < len(k) :
            if i + 1 < len(k) and k[i] == byte_tuple[0] and k[i+1] == byte_tuple[1]:
                new_k.append(b"".join(byte_tuple))
                i += 1
            else:
                new_k.append(k[i])
            i += 1
        new_pretok[tuple(new_k)] = v
    return new_pretok

def get_next_token(text, pattern: str | None=None):
    if pattern is None:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(pattern, text)


def find_chunk_boundaries(
    file: BinaryIO,
    split_special_token: bytes,
    target_chunk_bytes: int = 32,  # Default to 32 MB chunks
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently based on a target byte size.
    May return fewer chunks if boundaries overlap or the file is smaller than target_chunk_bytes.
    """
    target_chunk_bytes *= (1024 * 1024)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Edge case: Empty or tiny file
    if file_size == 0:
        return [0]
    if file_size <= target_chunk_bytes:
        return [0, file_size]

    # Calculate desired number of chunks dynamically from target chunk size
    desired_num_chunks = max(1, file_size // target_chunk_bytes)

    # Initial uniform chunk size guess
    chunk_size = file_size // desired_num_chunks

    # Initial boundary guesses
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead 4KB at a time to locate token

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, snap boundary to end of file
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break

            # Align boundary to the next occurrence of split_special_token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
                
            initial_position += mini_chunk_size

    # Remove duplicates (in case adjacent boundaries mapped to the same token) and return sorted
    return sorted(set(chunk_boundaries))

def get_pretok_bytes(pretok) -> dict[tuple[bytes, ...], int]:
    pretok_rep = {}
    for pt, count in pretok.items():
        pt_tuple = tuple(bytes([c]) for c in pt.encode("utf-8"))
        pretok_rep[pt_tuple] = count
    return pretok_rep

def process_chunk_range(boundary_pair: tuple[int, int], file_path: str, pattern: str | None) -> dict[str, int]:
    """Worker function executed in parallel processes."""
    start, end = boundary_pair
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    if pattern:
        chunks = re.split(pattern, chunk)
    else:
         chunks = [chunk]

    pretoken_count = collections.defaultdict(int)
    for text in chunks:
        for _match in get_next_token(text):
            pretoken = _match.group()
            pretoken_count[pretoken] += 1
    return pretoken_count

def batch_pretokens_generator(
    file_path: str, 
    boundaries: list[int], 
    pattern: str | None = None, 
    num_processes: int = 4,
) -> Iterator[dict[str, int]]:
    """Generates processed text segments in parallel."""
    # Build list of (start, end) tuple pairs
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    # Partial function to lock in static arguments (file path and pattern)
    worker_fn = functools.partial(process_chunk_range, file_path=file_path, pattern=pattern)
    
    with mp.Pool(processes=num_processes) as pool:
        # imap_unordered streams results as workers finish; 
        # use pool.imap if order preservation across chunk boundaries matters
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for pretoken_count in pool.imap_unordered(worker_fn, chunk_ranges):
                yield pretoken_count

def get_streaming_pretokens(path: str | os.PathLike, special_tokens: list[str]=["<|endoftext|>"], num_processes: int = 4, chunk_size_mb: int=32,) -> Iterator[dict[str, int]]:
    # read the text efficiently, breaking the long text on special_tokens
    special_token = b"<|endoftext|>"
    if special_tokens and "<|endoftext|>" not in special_tokens:
        # best effort
        special_token = special_tokens[0].encode("utf-8")
    
    with open(str(path), "rb") as f:
        # first split into parts using special token
        boundaries = find_chunk_boundaries(f, special_token, chunk_size_mb)
        # 1. Build the escaped regex pattern joined by '|'
        pattern = None
        if special_tokens:
            pattern = "|".join(map(re.escape, special_tokens))

        # send each start/end pair to a set of processes.
        return batch_pretokens_generator(
            str(path), boundaries, pattern, num_processes,
        )


if __name__ == "__main__":
    import time
    num_proc = os.cpu_count() or 4
    target_sizes = [4, 16, 32, 64]
    file_path = "data/owt_train.txt"
    start_time = time.perf_counter()
    for size_mb in target_sizes:
        get_streaming_pretokens(file_path, ["<|endoftext|>"], num_proc, size_mb)
        elapsed = time.perf_counter() - start_time
        print(f"Target Chunk Size: {size_mb} MB | Chunks: ... | Time: {elapsed:.2f}s")