import os

import numpy as np

from pathlib import Path


def make_dir_if_not_exist(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_filelines(path: Path) -> int:
    return sum([1 for _ in open(path)])


def pad_sequences(dataset, limit_len, pad_value=0):
    if isinstance(dataset[0], list):
        batch_size = len(dataset)
    else:
        batch_size = 1
        dataset = [dataset]

    padded_sequences = np.full((batch_size, limit_len), pad_value)

    for i, inst in enumerate(dataset):
        len_inst = len(inst)

        if len_inst > limit_len:
            padded_sequences[i, :] = np.array(inst[:limit_len])
        elif len_inst <= limit_len:
            padded_sequences[i, :len_inst] = np.array(inst[:len_inst])

    return padded_sequences

