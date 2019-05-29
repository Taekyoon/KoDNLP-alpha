from configs.constants import RANDOM_SEED

import torch
from tqdm import tqdm


class NERModelTrainer(object):
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

        self._metric_fn = metric_fn

    def train(self):
        torch.manual_seed(self._random_seed)
        for i in range(self._epochs):
            self._train_epoch(i)

    def _eval(self):
        total, score, f1_score = 0, 0., 0.

        self._model.eval()

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            input_batch = sampled_batch['inputs']['value']
            target_batch = sampled_batch['entities']

            score, tag_seq = self._model(input_batch)
            total += 1

        else:
            pass

        return score, f1_score / total

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        total, tr_loss = 0, 0

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            input_batch = torch.squeeze(sampled_batch['inputs']['value'], dim=1)
            target_batch = torch.squeeze(sampled_batch['entities'], dim=1)

            loss = self._model.neg_log_likelihood(input_batch, target_batch)
            tr_loss += loss
            self._backprop(loss)

            if (epoch * steps_in_epoch + (step + 1)) % self._eval_steps == 0:
                val_score, val_f1 = self._eval()
                self._model.train()
                print(tr_loss / (step+1))
                # tqdm.write('epoch : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                #                                                                                       tr_loss,
                #                                                                                       val_f1,
                #                                                                                       val_score))

    def _backprop(self, loss) -> None:
        optimizer = self._optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_model(self):
        pass
