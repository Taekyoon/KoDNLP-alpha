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

        self.word2idx = None
        self.idx2word = None

    def fit(self, tokenized_dataset: List) -> None:
        linear_dataset = self._square_to_linear(tokenized_dataset)
        max_size = self.max_size

        if self.word_frequency is None:
            self.word_frequency = Counter(linear_dataset)
        else:
            self.word_frequency.update(linear_dataset)

        self.word_frequency = self._filter_min_freq(self.word_frequency, self.min_freq)

        if len(self.word_frequency) < max_size:
            max_size = len(self.word_frequency)

        most_common_word_freq = self.word_frequency.most_common(max_size)
        self._create_word_dict()

        for word, _ in most_common_word_freq:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

        return

    def to_indices(self, tokens: List):
        pass

    def to_tokens(self, indices: List):
        pass

    def _create_word_dict(self) -> None:
        self.word2idx = {self.padding_token: 0,
                         self.unknown_token: 1}
        self.idx2word = {0: self.padding_token,
                         1: self.unknown_token}
        self.vocab_size += 2

        if self.bos_token is not None:
            self.word2idx[self.bos_token] = self.vocab_size
            self.idx2word[self.vocab_size] = self.bos_token
            self.vocab_size += 1

        if self.eos_token is not None:
            self.word2idx[self.eos_token] = self.vocab_size
            self.idx2word[self.vocab_size] = self.eos_token
            self.vocab_size += 1

        if self.reserved_tokens is not None:
            self.word2idx[self.reserved_tokens] = self.vocab_size
            self.idx2word[self.vocab_size] = self.reserved_tokens
            self.vocab_size += 1

        return

    @classmethod
    def _filter_min_freq(cls, word_frequency: Counter, min_freq: int) -> Counter:
        for word, freq in list(word_frequency.items()):
            if freq < min_freq:
                del word_frequency[word]

        return word_frequency

    @classmethod
    def _square_to_linear(cls, squared_list: List) -> List:
        return [word for sequence in squared_list for word in sequence]
