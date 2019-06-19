import torch
import numpy as np
from tqdm import tqdm

import logging
from prettytable import PrettyTable

from trainer.metrics import f1

logger = logging.getLogger(__name__)


class SLUModelEvaluator(object):
    def __init__(self,
                 model,
                 data_loader):
        self._model = model
        self._data_loader = data_loader

        self._device = torch.device('cpu')

        self._eval_score = None
        self._eval_tag_f1_score = None
        self._eval_class_f1_score = None

    def eval(self):
        logger.info('now evaluate!')

        self._model.eval()

        accumulated_tag_preds, accumulated_tag_targets = [], []
        accumulated_class_preds, accumulated_class_targets = [], []

        for step, sampled_batch in tqdm(enumerate(self._data_loader), desc='evaluation steps',
                                        total=len(self._data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['slots'].view(batch_size, -1).to(self._device)
            class_batch = sampled_batch['intents'].to(self._device)

            pred_score, tag_seq, class_prob = self._model(input_batch)

            accumulated_tag_preds.append(tag_seq.numpy())
            accumulated_tag_targets.append(target_batch.cpu().numpy())

            accumulated_class_preds.append(torch.argmax(class_prob, dim=-1).numpy())
            accumulated_class_targets.append(class_batch.numpy())
        else:
            tag_f1_score = f1(np.concatenate(accumulated_tag_preds, axis=None),
                              np.concatenate(accumulated_tag_targets, axis=None))
            class_f1_score = f1(np.concatenate(accumulated_class_preds, axis=None),
                                np.concatenate(accumulated_class_targets, axis=None))

        self._eval_tag_f1_score = tag_f1_score
        self._eval_class_f1_score = class_f1_score

        logger.info('evaluation done!')

    def summary(self):
        table = PrettyTable(['Name', 'Score'])
        table.add_row(['Tag F1 score', "{:.3f}".format(self._eval_tag_f1_score)])
        table.add_row(['Intent Acc score', "{:.3f}".format(self._eval_class_f1_score)])

        return table
