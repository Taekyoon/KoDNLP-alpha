from typing import Tuple
import json

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset

from data.utils import pad_sequences


class SLUDatasetFromJSONFile(Dataset):
    def __init__(self,
                 json_path: str,
                 enable_length: bool = True,
                 limit_pad_len: int = None,
                 pad_value: int = 0) -> None:
        dataset = json.load(open(json_path, 'rb'))

        self._inputs = np.array(dataset['inputs'])
        self._slots = np.array(dataset['slots'])
        self._intents = np.array(dataset['intents'])

        self.enable_length = enable_length
        self.limit_pad_len = limit_pad_len
        self.pad_value = pad_value

        return

    def __len__(self) -> int:
        len_dataset = len(self._inputs)

        return len_dataset

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sampled_instances = dict()
        sampled_instances['inputs'] = dict()

        sampled_inputs = self._inputs[idx]
        sampled_slots = self._slots[idx]
        sampled_intents = self._intents[idx]

        if self.enable_length:
            inputs_length = [len(inst) for inst in sampled_inputs]
            sampled_instances['inputs']['length'] = Tensor(inputs_length)

        if self.limit_pad_len is not None:
            sampled_inputs = pad_sequences(sampled_inputs, self.limit_pad_len, pad_value=self.pad_value)
            sampled_slots = pad_sequences(sampled_slots, self.limit_pad_len, pad_value=self.pad_value)

        sampled_instances['inputs']['value'] = Tensor(sampled_inputs)
        sampled_instances['slots'] = Tensor(sampled_slots)
        sampled_instances['intents'] = Tensor(sampled_intents)

        return sampled_instances


class NERDatasetFromJSONFile(Dataset):
    def __init__(self,
                 json_path: str,
                 enable_length: bool = True,
                 limit_pad_len: int = None,
                 pad_value: int = 0) -> None:
        dataset = json.load(open(json_path, 'rb'))

        self._inputs = dataset['inputs']
        self._entities = dataset['entities']

        self.enable_length = enable_length
        self.limit_pad_len = limit_pad_len
        self.pad_value = pad_value

        return

    def __len__(self) -> int:
        len_dataset = len(self._inputs)

        return len_dataset

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sampled_instances = dict()
        sampled_instances['inputs'] = dict()

        sampled_inputs = self._inputs[idx]
        sampled_entities = self._entities[idx]

        if self.enable_length:
            if isinstance(sampled_inputs[0], list):
                inputs_length = [
                    len(inst) if self.limit_pad_len is None and len(inst) < self.limit_pad_len else self.limit_pad_len
                    for inst in sampled_inputs]
                sampled_instances['inputs']['length'] = Tensor(inputs_length)
            else:
                inputs_length = [len(sampled_inputs) if self.limit_pad_len is None and len(
                    sampled_inputs) < self.limit_pad_len else self.limit_pad_len]
                sampled_instances['inputs']['length'] = Tensor(inputs_length).long()

        if self.limit_pad_len is not None:
            sampled_inputs = pad_sequences(sampled_inputs, self.limit_pad_len, self.pad_value)
            sampled_entities = pad_sequences(sampled_entities, self.limit_pad_len, self.pad_value)

        sampled_instances['inputs']['value'] = Tensor(sampled_inputs).long()
        sampled_instances['entities'] = Tensor(sampled_entities).long()

        return sampled_instances
