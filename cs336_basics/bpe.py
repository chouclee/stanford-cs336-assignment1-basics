import regex as re
import os

from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenization(input_path: str | os.PathLike,
                     start: int,
                     end: int,
                     special_tokens: list[str], queue: Queue):
    with open(input_path, mode="rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

        # Unfortunate step for Windows only!
        unix_style_chunk = chunk.replace(b"\r\n", b"\n")
        re_pattern = "|".join([re.escape(token) for token in special_tokens])
        documents = re.split(re_pattern, unix_style_chunk.decode("utf-8"))
        freq_table = {}
        for doc in documents:
            if len(doc) == 0:
                continue
            matches = re.finditer(PATTERN, doc)
            for match in matches:
                pre_token = match.group()
                if len(pre_token) == 0:
                    continue
                freq_table[pre_token] = freq_table.get(pre_token, 0) + 1
        queue.put(freq_table)


def pre_compute(freq_table: dict[str, int]):
    stats: dict[tuple, int] = {}
    pair_pre_token_reverse_index: dict[tuple, list[int]] = {}
    # Stores pre_token (list of ids) & its frequency (tuple) as list so we can use pair_pre_token_reverse_index to
    # quickly find out those pre tokens needs to be updated after merge
    freq_list = []
    for pre_token, freq in freq_table.items():
        ids = list(map(int, pre_token.encode("utf-8")))  # abc -> [97, 98, 99]
        freq_list.append((ids, freq))

    for index, (ids, freq) in enumerate(freq_list):
        for pair in zip(ids[:-1], ids[1:]):
            update_stats(stats, pair_pre_token_reverse_index, pair, freq, index)
    return stats, pair_pre_token_reverse_index, freq_list


def merge(
    stats: dict[tuple, int],
    pair_pre_token_reverse_index: dict[tuple, list[int]],
    pair,
    id,
    freq_list,
):
    for index in pair_pre_token_reverse_index.get(pair, []):
        ids, freq = freq_list[index]
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(id)
                # when old pair is replaced by the new id, update the count and reverse index map
                # e.g. "jabc"
                # when replacing "ab" with new id "x" (illustration purpose)
                # "jabc" -> "jxc"
                # We get new pairs "jx" and "xc", but we also lose some frequences for pair "ja" & "bc"
                if i > 0:
                    update_stats(
                        stats,
                        pair_pre_token_reverse_index,
                        (ids[i - 1], id),
                        freq,
                        index,
                    )
                    if stats.get((ids[i - 1], ids[i]), 0) > 0:
                        stats[(ids[i - 1], ids[i])] = (
                            stats.get((ids[i - 1], ids[i])) - freq
                        )
                if i < len(ids) - 2:
                    update_stats(
                        stats,
                        pair_pre_token_reverse_index,
                        (id, ids[i + 2]),
                        freq,
                        index,
                    )
                    if stats.get((ids[i + 1], ids[i + 2]), 0) > 0:
                        stats[(ids[i + 1], ids[i + 2])] = (
                            stats.get((ids[i + 1], ids[i + 2])) - freq
                        )
                i = i + 2
            else:
                new_ids.append(ids[i])
                i = i + 1
        freq_list[index] = (new_ids, freq)
    pair_pre_token_reverse_index.pop(pair)
    stats.pop(pair)


def update_stats(
    stats: dict[tuple, int],
    pair_pre_token_reverse_index: dict[tuple, list[int]],
    pair,
    freq,
    index,
):
    stats[pair] = stats.get(pair, 0) + freq
    reverse_index = pair_pre_token_reverse_index.get(pair, [])
    (
        reverse_index.append(index)
        if len(reverse_index) == 0 or reverse_index[-1] != index
        else None
    )
    pair_pre_token_reverse_index[pair] = reverse_index


def merges(freq_table: dict[str, int], vocab_size, special_tokens: list[str]):
    num_merges = vocab_size - 256 - len(special_tokens)
    stats, pair_pre_token_reverse_index, freq_list = pre_compute(freq_table)
    # merge_dict = {}  # (int, int) -> index
    merges = []
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i in range(num_merges):
        pair, freq = max(
            stats.items(),
            key=lambda item: (item[1], [(vocab[item[0][0]], vocab[item[0][1]])]),
        )
        if freq == 1:
            break
        new_id = 256 + i
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        # print(f"merging {pair[0]}:{vocab[pair[0]]} & {pair[1]}:{vocab[pair[1]]}")
        merge(stats, pair_pre_token_reverse_index, pair, new_id, freq_list)
        # merge_dict[pair] = new_id
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        # c = collections.Counter(stats)
        # for k, v in c.most_common(15):
        #     print(f"{k[0]}, {k[1]}, freq: {v}")
        #     print(f"{(vocab[k[0]] + vocab[k[1]]).decode('utf-8')}, freq: {v}")
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes,
):
    with open(input_path, mode="rb") as f:
        processes = []
        queue = Queue()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    freq_table = {}
    
    # override the process number if find_chunk_boundaries may return less than 
    # requested chunks
    num_processes = len(boundaries[:-1])  
    
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        proc = Process(
            target=pre_tokenization,
            args=(input_path, start, end, special_tokens, queue),
        )
        processes.append(proc)
        proc.start()

    for _ in range(num_processes):
        table = queue.get()
        # merge the table
        for key, val in table.items():
            freq_table[key] = freq_table.get(key, 0) + val

    return merges(freq_table, vocab_size, special_tokens)
