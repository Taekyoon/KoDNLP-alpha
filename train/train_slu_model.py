from configs.constants import RANDOM_SEED

import torch
import numpy as np
from tqdm import tqdm

from train.metrics import f1


class SLUModelTrainer(object):
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
                                                                  and gpu_device > 0 else torch.device('cpu')

        self.best_tag_val_f1_score = 0.
        self.best_class_val_f1_score = 0.

        self._metric_fn = metric_fn

    def train(self):
        torch.manual_seed(self._random_seed)
        for i in range(self._epochs):
            self._train_epoch(i)

    def _eval(self):
        score, f1_score = 0., 0.

        self._model.eval()

        accumulated_tag_preds, accumulated_tag_targets = [], []
        accumulated_class_preds, accumulated_class_targets = [], []

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            input_batch = torch.squeeze(sampled_batch['inputs']['value'], dim=1)
            target_batch = torch.squeeze(sampled_batch['slots'], dim=1)
            class_batch = sampled_batch['intents']

            pred_score, tag_seq, class_prob = self._model(input_batch)
            score += torch.mean(pred_score)

            accumulated_tag_preds.append(tag_seq.numpy())
            accumulated_tag_targets.append(target_batch.numpy())

            accumulated_class_preds.append(torch.argmax(class_prob, dim=-1).numpy())
            accumulated_class_targets.append(class_batch.numpy())
        else:
            score = score / (step + 1)
            tag_f1_score = f1(np.array(accumulated_tag_preds), np.array(accumulated_tag_targets))
            class_f1_score = f1(np.array(accumulated_class_preds), np.array(accumulated_class_targets))

        return score, tag_f1_score, class_f1_score

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        total, tr_loss, tr_tag_loss, tr_class_loss = 0, 0., 0., 0.

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            total_steps = epoch * steps_in_epoch + (step + 1)

            input_batch = torch.squeeze(sampled_batch['inputs']['value'], dim=1)
            target_batch = torch.squeeze(sampled_batch['slots'], dim=1)
            class_batch = sampled_batch['intents']

            tag_loss, class_loss = self._model.neg_log_likelihood(input_batch, target_batch, class_batch)

            tr_tag_loss += tag_loss
            tr_class_loss += class_loss

            loss = tag_loss + class_loss
            tr_loss += loss
            self._backprop(loss)

            if total_steps % self._eval_steps == 0:
                val_score, val_tag_f1, val_class_f1 = self._eval()
                self.best_tag_val_f1_score = val_tag_f1 if val_tag_f1 > self.best_tag_val_f1_score \
                    else self.best_tag_val_f1_score
                self.best_class_val_f1_score = val_class_f1 if val_class_f1 > self.best_class_val_f1_score \
                    else self.best_class_val_f1_score
                self._model.train()
                tqdm.write(
                    'epoch : {}, steps : {}, tr_loss : {:.3f}, tr_tag_loss : {:.3f}, tr_class_loss : {:.3f}, '
                    'val_tag_f1 : {:.3f}, val_score : {:.3f}, val_class_f1 : {:.3f}'.format(epoch + 1,
                                                                                            total_steps,
                                                                                            tr_loss / (step + 1),
                                                                                            tr_tag_loss / (step + 1),
                                                                                            tr_class_loss / (step + 1),
                                                                                            val_tag_f1,
                                                                                            val_score,
                                                                                            val_class_f1))

    def _backprop(self, loss) -> None:
        optimizer = self._optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_model(self):
        pass
