from typing import List
from collections import Counter


class Vocabulary(object):
    def __init__(self,
                 vocab_size=1000):
        self.vocab_size = vocab_size

        self.data_frequency = None

        self.word2idx = None
        self.idx2word = None

    def update(self, tokenized_dataset: List):
        pass

    def fit(self, tokenized_dataset: List):
        linear_dataset = self._square_to_linear(tokenized_dataset)

        if self.data_frequency is None:
            self.data_frequency = Counter(linear_dataset)
        else:
            self.data_frequency.update(linear_dataset)

    @classmethod
    def _square_to_linear(cls, squared_list):
        return [word for sequence in squared_list for word in sequence]
