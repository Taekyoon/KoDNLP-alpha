import copy

from typing import List
from collections import Counter


class Vocabulary(object):
    def __init__(self,
                 max_size=None,
                 min_freq=1,
                 unknown_token='<unk>',
                 padding_token='<pad>',
                 bos_token='<bos>',
                 eos_token='<eos>',
                 reserved_tokens=None):

        self.max_size = max_size
        self.min_freq = min_freq
        self.vocab_size = 0

        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.reserved_tokens = reserved_tokens

        self.word_frequency = None

        self._word_to_idx = None
        self._idx_to_word = None

    def fit(self, tokenized_dataset: List) -> None:
        if isinstance(tokenized_dataset[0], list):
            linear_dataset = self._square_to_linear(tokenized_dataset)
        else:
            linear_dataset = tokenized_dataset

        max_size = self.max_size

        if self.word_frequency is None:
            self.word_frequency = Counter(linear_dataset)
        else:
            self.word_frequency.update(linear_dataset)

        filtered_word_frequency = self._filter_min_freq(self.word_frequency, self.min_freq)

        if max_size is None or len(filtered_word_frequency) < max_size:
            max_size = len(filtered_word_frequency)

        most_common_word_freq = filtered_word_frequency.most_common(max_size)
        self._create_word_dict()

        for word, _ in most_common_word_freq:
            self._idx_to_word.append(word)
            self._word_to_idx[word] = len(self._idx_to_word) - 1

        return

    def to_indices(self, tokens: List):
        return self[tokens]

    def to_tokens(self, indices: List):
        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_word) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indices` is invalid.'.format(idx))
            else:
                tokens.append(self._idx_to_word[idx])

        return tokens[0] if to_reduce else tokens

    def to_json(self, json_path):
        pass

    @classmethod
    def from_json(cls, json_path):
        pass

    @property
    def word_to_idx(self):
        return self._word_to_idx

    @property
    def idx_to_word(self):
        return self._idx_to_word

    def __len__(self):
        return len(self._idx_to_word)

    def __getitem__(self, words):
        if not isinstance(words, (list, tuple)):
            return self._word_to_idx[words] if words in self._word_to_idx else self._word_to_idx[self.unknown_token]
        else:
            return [self._word_to_idx[w] if w in self._word_to_idx else self._word_to_idx[self.unknown_token]
                    for w in words]

    def _create_word_dict(self) -> None:
        self._word_to_idx = dict()
        self._idx_to_word = list()

        if self.padding_token is not None:
            self._idx_to_word.append(self.padding_token)
            self._word_to_idx[self.padding_token] = len(self._idx_to_word) - 1

        if self.unknown_token is not None:
            self._idx_to_word.append(self.unknown_token)
            self._word_to_idx[self.unknown_token] = len(self._idx_to_word) - 1

        if self.bos_token is not None:
            self._idx_to_word.append(self.bos_token)
            self._word_to_idx[self.bos_token] = len(self._idx_to_word) - 1

        if self.eos_token is not None:
            self._idx_to_word.append(self.eos_token)
            self._word_to_idx[self.eos_token] = len(self._idx_to_word) - 1

        if self.reserved_tokens is not None:
            for token in self.reserved_tokens:
                self._idx_to_word.append(token)
                self._word_to_idx[token] = len(self._idx_to_word) - 1

        return

    @classmethod
    def _filter_min_freq(cls, word_frequency: Counter, min_freq: int) -> Counter:
        filtered_word_frequency = copy.deepcopy(word_frequency)

        for word, freq in list(filtered_word_frequency.items()):
            if freq < min_freq:
                del filtered_word_frequency[word]

        return filtered_word_frequency

    @classmethod
    def _square_to_linear(cls, squared_list: List) -> List:
        return [word for sequence in squared_list for word in sequence]
