from configs.constants import RANDOM_SEED

import torch
import numpy as np
from tqdm import tqdm

from train.trainer import Trainer
from train.metrics import f1


class NERModelTrainer(Trainer):
    def __init__(self,
                 train_data_loader,
                 valid_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 learning_rate=1e-4,
                 optimizer=torch.optim.Adam,
                 metric_fn=None,
                 gpu_device=-1,
                 random_seed=RANDOM_SEED):

        self._epochs = epochs
        self._eval_steps = eval_steps
        self._learning_rate = learning_rate
        self._random_seed = random_seed

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

        self.best_val_f1_score = 0.

        self._metric_fn = metric_fn

    def _eval(self):
        score, f1_score = 0., 0.

        self._model.eval()

        accumulated_preds, accumulated_targets = [], []

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            pred_score, tag_seq = self._model(input_batch)
            score += torch.mean(pred_score)
            accumulated_preds.append(tag_seq.cpu().numpy())
            accumulated_targets.append(target_batch.cpu().numpy())
        else:
            score = score / (step + 1)
            f1_score = f1(np.concatenate(accumulated_preds, axis=None), np.concatenate(accumulated_targets, axis=None))

        return score, f1_score

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        total, tr_loss = 0, 0

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)
            total_steps = epoch * steps_in_epoch + (step + 1)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            loss = self._model.neg_log_likelihood(input_batch, target_batch)
            tr_loss += loss
            self._backprop(loss, self._optimizer)

            if total_steps % self._eval_steps == 0:
                val_score, val_f1 = self._eval()
                self.best_val_f1_score = val_f1 if val_f1 > self.best_val_f1_score else self.best_val_f1_score
                self._model.train()
                tqdm.write(
                    'epoch : {}, steps : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                                                                                                           total_steps,
                                                                                                           tr_loss / (step + 1),
                                                                                                           val_f1,
                                                                                                           val_score))
