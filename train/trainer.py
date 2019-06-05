import torch


class Trainer(object):
    def train(self):
        torch.manual_seed(self._random_seed)
        for i in range(self._epochs):
            self._train_epoch(i)

    def _backprop(self, loss, optimizer) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def _train_epoch(self, epoch):
        raise NotImplementedError()
