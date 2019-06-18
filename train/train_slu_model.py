from train.trainer import Trainer
from configs.constants import RANDOM_SEED

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path

import logging

from train.metrics import f1, acc

logger = logging.getLogger(__name__)


class SLUModelTrainer(Trainer):
    def __init__(self,
                 train_data_loader,
                 valid_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 deploy_path=Path('./tmp'),
                 learning_rate=3e-4,
                 optimizer=torch.optim.Adam,
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
        self._tb_writer = SummaryWriter(self._deploy_path / 'logs')

        if self._device.type == 'cpu':
            torch.manual_seed(self._random_seed)
        else:
            torch.cuda.manual_seed_all(self._random_seed)

        self.train_loss = -1
        self.best_tag_val_f1_score = 0.
        self.best_class_val_acc_score = 0.

        logger.info('deploy path: {}'.format(self._deploy_path))
        logger.info('random seed number: {}'.format(self._random_seed))
        logger.info('learning rate: {}'.format(self._learning_rate))
        logger.info('evaluation check steps: {}'.format(self._eval_steps))
        logger.info('number of epochs: {}'.format(self._epochs))
        logger.info('training device: {}'.format(self._device.type))

    def _eval(self):
        score = 0.

        logger.info('now evaluating...')

        self._model.eval()

        accumulated_tag_preds, accumulated_tag_targets = [], []
        accumulated_class_preds, accumulated_class_targets = [], []

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['slots'].view(batch_size, -1).to(self._device)
            class_batch = sampled_batch['intents'].to(self._device)

            pred_score, tag_seq, class_prob = self._model(input_batch)
            score += torch.mean(pred_score)

            accumulated_tag_preds.append(tag_seq.cpu().numpy())
            accumulated_tag_targets.append(target_batch.cpu().numpy())

            accumulated_class_preds.append(torch.argmax(class_prob, dim=-1).cpu().numpy())
            accumulated_class_targets.append(class_batch.cpu().numpy())
        else:
            score = score / (step + 1)
            tag_f1_score = f1(np.concatenate(accumulated_tag_preds, axis=None),
                              np.concatenate(accumulated_tag_targets, axis=None))
            class_acc_score = acc(np.concatenate(accumulated_class_preds, axis=None),
                                  np.concatenate(accumulated_class_targets, axis=None))

        return score, tag_f1_score, class_acc_score

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        total, tr_loss, tr_tag_loss, tr_class_loss = 0, 0., 0., 0.

        logging.info('start training epoch {}'.format(epoch + 1))

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            total_steps = epoch * steps_in_epoch + (step + 1)
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['slots'].view(batch_size, -1).to(self._device)
            class_batch = sampled_batch['intents'].to(self._device)
            input_len_batch = sampled_batch['inputs']['length'].view(batch_size)

            tag_loss, class_loss = self._model.neg_log_likelihood(input_batch, target_batch, class_batch,
                                                                  input_len_batch)

            tr_tag_loss += tag_loss
            tr_class_loss += class_loss

            loss = tag_loss + class_loss
            tr_loss += loss
            self._backprop(loss, self._optimizer)

            if self._eval_steps > 0 and total_steps % self._eval_steps == 0:
                val_score, val_tag_f1, val_class_acc = self._eval()

                if val_class_acc >= self.best_class_val_acc_score and val_tag_f1 >= self.best_tag_val_f1_score:
                    self.best_class_val_acc_score = val_class_acc
                    self.best_tag_val_f1_score = val_tag_f1

                    self._save_model(self._model, self._deploy_path / 'best_val.pkl')

                self._model.train()
                logger.info(
                    'epoch : {}, steps : {}, tr_loss : {:.3f}, tr_tag_loss : {:.3f}, tr_class_loss : {:.3f}, '
                    'val_tag_f1 : {:.3f}, val_score : {:.3f}, val_class_acc : {:.3f}'.format(epoch + 1,
                                                                                             total_steps,
                                                                                             tr_loss / (step + 1),
                                                                                             tr_tag_loss / (step + 1),
                                                                                             tr_class_loss / (step + 1),
                                                                                             val_tag_f1,
                                                                                             val_score,
                                                                                             val_class_acc))
                self._tb_writer.add_scalar('train_loss', tr_loss / (step + 1), total_steps)
                self._tb_writer.add_scalar('train_tag_loss', tr_tag_loss / (step + 1), total_steps)
                self._tb_writer.add_scalar('train_class_loss', tr_class_loss / (step + 1), total_steps)

                self._tb_writer.add_scalar('valid_tag_f1', val_tag_f1, total_steps)
                self._tb_writer.add_scalar('valid_class_acc', val_class_acc, total_steps)

                filename = 'checkpoint_' + str(total_steps) + '_model.pkl'
                self._save_model(self._model, self._deploy_path / 'checkpoint' / filename)

        else:
            logger.info('epoch {} is done!'.format(epoch + 1))
            val_score, val_tag_f1, val_class_acc = self._eval()

            if val_class_acc >= self.best_class_val_acc_score and val_tag_f1 >= self.best_tag_val_f1_score:
                self.best_class_val_acc_score = val_class_acc
                self.best_tag_val_f1_score = val_tag_f1

                self._save_model(self._model, self._deploy_path / 'best_val.pkl')

            logger.info(
                'epoch : {}, steps : {}, tr_loss : {:.3f}, tr_tag_loss : {:.3f}, tr_class_loss : {:.3f}, '
                'val_tag_f1 : {:.3f}, val_score : {:.3f}, val_class_acc : {:.3f}'.format(epoch + 1,
                                                                                         total_steps,
                                                                                         tr_loss / (step + 1),
                                                                                         tr_tag_loss / (step + 1),
                                                                                         tr_class_loss / (step + 1),
                                                                                         val_tag_f1,
                                                                                         val_score,
                                                                                         val_class_acc))
            logger.info('current best tag f1: {:.3f}'.format(self.best_tag_val_f1_score))
            logger.info('current best class acc: {:.3f}'.format(self.best_class_val_acc_score))

            return tr_loss / (step + 1)
