import regex as re
def get_next_token(text):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)

# if pattern:
#     chunks = re.split(pattern, chunk)
# else:
#     chunks = [chunk]
pattern = "|".join(map(re.escape, ["<|endoftext|>", "<|eott|>"]))

def get_next_token_load(text, pattern: str | None=None):
    
    base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if pattern:
        combined_pattern = rf"(?:{pattern})|(?:{base_pattern})"
    else:
        combined_pattern = base_pattern

    compiled_regex = re.compile(combined_pattern)
    return re.finditer(compiled_regex, text)

if __name__ == "__main__":
    # <|endoftext|>
    text = "The boy has things to do \n\n\n\n"
    # chunks = re.split(pattern, text)
    # for chunk in chunks:
    # print(list(_match.group() for _match in get_next_token(text)))
    special = ["<|special_tokens|>", "<||>"]

    pattern = "|".join(map(re.escape, special))
    # print(list(get_next_token_load("None of it <|special_tokens|>", pattern)))

    curr_text = "<|special_tokens|>"
    match = re.search(r'^(.*?)(\s*)$', curr_text, flags=re.DOTALL)
    curr_text, trailing = match.groups()
    # print("curr_text", curr_text)
    # print("next", trailing)

    special_token_split_pattern = f"({'|'.join(map(re.escape, special))})"
    print(re.split(special_token_split_pattern, curr_text))
