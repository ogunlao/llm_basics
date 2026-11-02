import json
import regex as re
from llm_basics.text_loading import get_next_token
from typing import Iterator, Iterable

def load_vocab(path: str, encoding="utf-8",) -> dict[int, bytes]:
    with open(path, "r", encoding=encoding,) as f:
        loaded_json = json.load(f)
        return {int(k): v.encode("latin-1") for k, v in loaded_json.items()}

def load_merges(path: str, encoding="utf-8",) -> list[tuple[bytes, bytes]]:
    with open(path, "r", encoding=encoding) as f:
        loaded = json.load(f)
        return [(tk1.encode("latin-1"), tk2.encode("latin-1")) for tk1, tk2 in loaded]

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Met à jour et marque comme récemment utilisée
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Éviction du premier élément (le moins récemment utilisé) si dépassement
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        """Construct a tokenizer from a given 
vocabulary, list of merges, and (optionally) a list of special tokens."""
        self.vocab = dict(vocab)
        self.merges = {merge: idx for idx, merge in enumerate(merges)}
        self.vocab2id = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens
        self.add_special_tokens(special_tokens)
        self.special_token_split_pattern = None
        if self.special_tokens:
            # Sort to cater for overlapping tokens
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_token_split_pattern = f"({'|'.join(map(re.escape, self.special_tokens))})"
        self.token_cache = LRUCache(200000)

    def add_special_tokens(self, special_tokens) -> None:
        # Add special tokens to the vocabulary if they aren’t already
        if special_tokens:
            last_idx = max(self.vocab.keys()) + 1
            for token in special_tokens:
                enc_token = token.encode("utf-8")
                if enc_token not in self.vocab2id:
                    self.vocab[last_idx] = enc_token
                    self.vocab2id[enc_token] = last_idx
                    last_idx += 1
        return

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        """Constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. """
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        
        return cls(vocab, merges, special_tokens)

    def get_merges(self, pretok:  list[tuple[bytes, ...]]) -> set[tuple[bytes, bytes]]:
        merges = set()
        for k in pretok:
            if len(k) < 1:
                continue
            for tk1, tk2 in zip(k, k[1:]):
                merges.add((tk1, tk2))
        return merges

    def replace_pretok(self, curr_pretok: list, byte_tuple: tuple[bytes, bytes]) -> list[tuple[bytes, ...]]:
        new_pretok = []
        for k in curr_pretok:
            new_k = []
            i = 0
            while i < len(k) :
                if i + 1 < len(k) and k[i] == byte_tuple[0] and k[i+1] == byte_tuple[1]:
                    new_k.append(b"".join(byte_tuple))
                    i += 1
                else:
                    new_k.append(k[i])
                i += 1
            new_pretok.append(tuple(new_k))
        return new_pretok

    def encode_pretoken(self, pretoken: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        cache = self.token_cache.get(pretoken) 
        if cache is not None:
            return list(cache)
        
        pt_tuple = tuple(bytes([c]) for c in pretoken.encode("utf-8"))
        pre_tokens_bytes = [pt_tuple]
        # Now convert bytes into int
        while True:
            merges = self.get_merges(pre_tokens_bytes)
            if not merges:
                break
            merge_tuple = min(merges, key=lambda x: self.merges.get(x, float("inf")))
            if merge_tuple not in self.merges:
                break
            pre_tokens_bytes = self.replace_pretok(pre_tokens_bytes, merge_tuple)

        byte_tuple = pre_tokens_bytes[0]
        idxs = [self.vocab2id[byte_str] for byte_str in byte_tuple]
        self.token_cache.put(pretoken, tuple(idxs))
        return idxs

    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        
        # first splip into groups without special tokens
        if self.special_token_split_pattern is not None:
            chunks = re.split(self.special_token_split_pattern, text)
        else:
            chunks = [text]

        # then pretokenise each group
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                idx = self.vocab2id.get(chunk.encode("utf-8"), None)
                if idx is not None:
                    tokens.append(idx)
                    continue

            pre_tokens: list[str] = [_match.group() for _match in get_next_token(chunk)]
            for pt in pre_tokens:
                idxs = self.encode_pretoken(pt)
                tokens.extend(idxs)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), returns a generator that lazily yields token IDs."""

        trailing = ""
        for text in iterable:
            _curr_text = trailing + text

            # find the first training whitespace
            curr_text = _curr_text.rstrip()
            trailing = _curr_text[len(curr_text):]
            if curr_text == "":
                continue
            yield from self.encode(curr_text)
        else:
            # iterable finished
            if trailing == '':
                return
            yield from self.encode(trailing)
        return

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        decoded = b"".join(self.vocab[idx] for idx in ids)
        return decoded.decode("utf-8", errors="replace")

    
if __name__ == "__main__":
    vocab_filepath = "data/tinystories_vocab.json"
    merges_filepath = "data/tinystories_merges.json"

    bpe_tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>", "<|pad|>"])
    text = ""
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
        # lines = f.readlines()
        i = 0
        for line in f:
            # all_lines.append(line)
            text += line
            i += 1

            if i == 10000: break
    # bpe_tokenizer.encode(text)
    assert text == bpe_tokenizer.decode(bpe_tokenizer.encode(text))
    all_lines = []
    lines = ""
    with open("data/sample_text.txt") as f:
        # lines = f.readlines()
        for line in f:
            all_lines.append(line)
            lines += line
    assert bpe_tokenizer.encode(lines) == list(x for x in bpe_tokenizer.encode_iterable(all_lines))
