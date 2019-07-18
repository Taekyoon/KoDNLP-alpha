import torch
from tqdm import tqdm

import logging
from prettytable import PrettyTable

from trainer.metrics import bleu

logger = logging.getLogger(__name__)


class TranslationModelEvaluator(object):
    def __init__(self,
                 model,
                 data_loader):
        self._model = model
        self._data_loader = data_loader

        self._device = torch.device('cpu')

        self._bleu_score = 0.

    def eval(self):
        logger.info('now evaluate!')

        self._model.eval()

        bleu_score = 0.

        for step, sampled_batch in tqdm(enumerate(self._data_loader), desc='evaluation steps',
                                        total=len(self._data_loader)):
            batch_size = sampled_batch['sources']['value'].size(0)

            source_batch = sampled_batch['sources']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['targets']['value'].view(batch_size, -1).to(self._device)
            predited_batch = self._model(source_batch)

            bleu_score += bleu(target_batch.tolist(), predited_batch.tolist())
        else:
            bleu_score /= (step + 1)

        self._bleu_score = bleu_score

        logger.info('evaluation done!')

    def summary(self):
        table = PrettyTable(['Name', 'Score'])
        table.add_row(['BLEU score', "{:.3f}".format(self._bleu_score)])

        return table
