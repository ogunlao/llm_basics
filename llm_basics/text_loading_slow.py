import os
from typing import BinaryIO
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def read_paragraph(path:  str | os.PathLike, special_tokens: list[str]=["<|endoftext|>"], num_processes: int = 4):
    # read the text efficiently, breaking the long text on special_tokens
    special_token = b"<|endoftext|>"
    if special_tokens and "<|endoftext|>" not in special_tokens:
        # best effort
        special_token = special_tokens[0].encode("utf-8")
    
    with open(str(path), "rb") as f:
        # first split into parts using special token
        boundaries = find_chunk_boundaries(f, num_processes, special_token)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # 1. Build the escaped regex pattern joined by '|'
        pattern = None
        if special_tokens:
            pattern = "|".join(map(re.escape, special_tokens))
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if pattern:
                texts = re.split(pattern, chunk)
            else:
                texts = [chunk]
            for text in texts:
                yield text
