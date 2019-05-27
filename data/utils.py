import numpy as np


def pad_sequences(dataset, limit_len, pad_value=0):
    batch_size = dataset.shape[0]

    padded_sequences = np.full((batch_size, limit_len), pad_value)

    for i, inst in enumerate(dataset):
        len_inst = len(inst)

        if len_inst > limit_len:
            padded_sequences[i, :] = inst[:limit_len]
        elif len_inst <= limit_len:
            padded_sequences[i, :len_inst] = inst[:len_inst]

    return padded_sequences

