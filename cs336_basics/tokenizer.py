from collections.abc import Iterable
from typing import Iterator
import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
def merge(ids, pair, idx):
  new_ids = []
  i = 0
  while i < len(ids):
    if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
      new_ids.append(idx)
      i += 2
    else:
      new_ids.append(ids[i])
      i += 1
  return new_ids
    
class Tokenizer:

    id_merges: dict[tuple[int, int], int]
    vocab_reverse: dict[bytes, int]
    special_tokens_pattern: re.Pattern | None

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.id_merges = {}
        for merge in merges:
            self.id_merges[
                (self.vocab_reverse[merge[0]], self.vocab_reverse[merge[1]])
            ] = self.vocab_reverse[merge[0] + merge[1]]

        if special_tokens is not None:
            # sort it by length so we match <|endoftext|><|endoftext|> before <|endoftext|>
            # in case there is overlapping special tokens
            special_tokens.sort(key=len, reverse=True)
            self.special_tokens = special_tokens
            self.special_tokens_pattern = re.compile(f"({'|'.join([re.escape(token) for token in special_tokens])})")
        else:
            self.special_tokens = []
            self.special_tokens_pattern = None


    def encode(self, text: str) -> list[int]:
        encoded_tokens = []
        chunks = []
        if self.special_tokens_pattern is not None:
            chunks = self.special_tokens_pattern.split(text)
        else:
            chunks = [text]
        for chunk in chunks:
            if chunk in self.special_tokens:
                encoded_tokens.append(self.vocab_reverse[chunk.encode("utf-8")])
                continue

            matches = re.finditer(PATTERN, chunk)
            for match in matches:
                pre_token = match.group()
                ids = [self.vocab_reverse[bytes([b])] for b in pre_token.encode("utf-8")]
                while len(ids) >= 2:
                    stats = get_stats(ids)
                    pair = min(stats, key=lambda p: self.id_merges.get(p, float("inf")))
                    if pair not in self.id_merges:
                        break
                    ids = merge(ids, pair, self.id_merges[pair])
                encoded_tokens.extend(ids)
        return encoded_tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
           for token in self.encode(line):
               yield token
               
    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode(encoding="utf-8", errors="replace")