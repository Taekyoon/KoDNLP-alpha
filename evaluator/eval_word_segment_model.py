import torch
from torch import nn

import jiwer
from tqdm import tqdm
from typing import List

import logging
from prettytable import PrettyTable

from data_manager.vocab import Vocabulary
from prepro.common import unspacing, text_to_list
from postpro.word_segment import segment_word_by_tags

from time import sleep

logger = logging.getLogger(__name__)


class WordSegmentModelEvaluator(object):
    def __init__(self,
                 model: nn.Module,
                 dataset: List,
                 input_vocab: Vocabulary,
                 tag_vocab: Vocabulary):
        self._model = model
        self._input_vocab = input_vocab
        self._tag_vocab = tag_vocab

        self._dataset = dataset

        self._device = torch.device('cpu')

        self._wer_score = None

    def eval(self):
        logger.info('now evaluate!')

        self._model.eval()

        wer_score = 0.
        score_failure_cnt = 0

        for step, text in tqdm(enumerate(self._dataset), desc='evaluation steps', total=len(self._dataset)):
            try:
                unspaced_text = unspacing(text.strip())
                tokenized_text = text_to_list(unspaced_text)

                input_batch = torch.Tensor([self._input_vocab.to_indices(tokenized_text)]).long()

                _, tag_seq = self._model(input_batch)
                labeled_tag_seq = self._tag_vocab.to_tokens(tag_seq[0].tolist())
                pred_text = segment_word_by_tags(unspaced_text, labeled_tag_seq)
                # print('org: ', text.strip())
                # print('prd: ', pred_text.strip())
                wer_score += jiwer.wer(text.strip(), pred_text.strip())
            except Exception as e:
                score_failure_cnt += 1
                logger.warning("Error message while calculating wer score: {}".format(e))
                logger.info('wer score failure {} times'.format(score_failure_cnt))
                raise ValueError()
            # sleep(0.7)

        else:
            wer_score = wer_score / (step + 1 - score_failure_cnt)

        self._wer_score = wer_score

        logger.info('evaluation done!')

    def summary(self):
        table = PrettyTable(['Name', 'Score'])
        table.add_row(['WER score', "{:.3f}".format(self._wer_score)])

        return table
