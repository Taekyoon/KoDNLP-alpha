from configs.constants import RANDOM_SEED

import logging
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from train.trainer import Trainer
from train.metrics import f1

logger = logging.getLogger(__name__)


class NERModelTrainer(Trainer):
    def __init__(self,
                 train_data_loader,
                 valid_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 deploy_path=Path('./tmp'),
                 learning_rate=3e-4,
                 optimizer=torch.optim.Adam,
                 metric_fn=None,
                 gpu_device=-1,
                 random_seed=RANDOM_SEED):

        self._epochs = epochs
        self._eval_steps = eval_steps
        self._learning_rate = learning_rate
        self._random_seed = random_seed

        self._deploy_path = deploy_path

        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._model = model
        self._optimizer = optimizer(params=self._model.parameters(), lr=self._learning_rate)

        self._device = torch.device('cuda:' + str(gpu_device)) if torch.cuda.is_available() \
                                                                  and gpu_device >= 0 else torch.device('cpu')

        if self._device.type == 'cpu':
            torch.manual_seed(self._random_seed)
        else:
            torch.cuda.manual_seed_all(self._random_seed)

        self.train_loss = -1
        self.best_val_f1_score = 0.

        self._metric_fn = metric_fn

        logger.info('deploy path: {}'.format(self._deploy_path))
        logger.info('random seed number: {}'.format(self._random_seed))
        logger.info('learning rate: {}'.format(self._learning_rate))
        logger.info('evaluation check steps: {}'.format(self._eval_steps))
        logger.info('number of epochs: {}'.format(self._epochs))
        logger.info('training device: {}'.format(self._device.type))

    def _eval(self):
        score, f1_score = 0., 0.

        logger.info('now evaluating...')

        self._model.eval()

        # accumulated_preds, accumulated_targets = [], []

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            pred_score, tag_seq = self._model(input_batch)
            score += pred_score.item()
            f1_score += f1(tag_seq.detach().numpy(), target_batch.detach().numpy())
            # accumulated_preds.append(tag_seq.cpu().detach().numpy())
            # accumulated_targets.append(target_batch.cpu().detach().numpy())
        else:
            score = score / (step + 1)
            # f1_score = f1(np.concatenate(accumulated_preds, axis=None), np.concatenate(accumulated_targets, axis=None))
            f1_score = f1_score / (step + 1)

        return score, f1_score

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        total, tr_loss = 0, 0

        logger.info('start training epoch {}'.format(epoch + 1))

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)
            total_steps = epoch * steps_in_epoch + (step + 1)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            loss = self._model.loss(input_batch, target_batch)
            tr_loss += loss
            self._backprop(loss, self._optimizer)

            if self._eval_steps > 0 and total_steps % self._eval_steps == 0:
                val_score, val_f1 = self._eval()

                if val_f1 > self.best_val_f1_score:
                    self.best_val_f1_score = val_f1

                    self._save_model(self._model, self._deploy_path / 'best_val.pkl')

                self._model.train()
                logger.info(
                    'epoch : {}, steps : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                                                                                                           total_steps,
                                                                                                           tr_loss / (step + 1),
                                                                                                           val_f1,
                                                                                                           val_score.item()))
                filename = 'checkpoint_' + str(total_steps) + '_model.pkl'
                self._save_model(self._model, self._deploy_path / 'checkpoint' / filename)
        else:
            logger.info('epoch {} is done!'.format(epoch + 1))
            val_score, val_f1 = self._eval()

            if val_f1 > self.best_val_f1_score:
                self.best_val_f1_score = val_f1

                self._save_model(self._model, self._deploy_path / 'best_val.pkl')

            self._model.train()
            logger.info(
                'epoch : {}, steps : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                                                                                                       total_steps,
                                                                                                       tr_loss / (step + 1),
                                                                                                       val_f1,
                                                                                                       val_score.item()))
            logger.info('current best tag f1: {:.3f}'.format(self.best_val_f1_score))

            return tr_loss / (step + 1)
