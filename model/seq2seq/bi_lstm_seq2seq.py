import torch
from torch import nn
from torch.nn import Embedding, Linear, CrossEntropyLoss

from model.modules.rnn import BiLSTM, BiLSTMCell


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_idx=0, num_layers=1, dropout=0.):
        super(BiLSTMEncoder, self).__init__()
        self.pad_idx = pad_idx

        self.embedding = Embedding(vocab_size, embedding_size, padding_idx=self.pad_idx)
        self.encoder = BiLSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, inputs: torch.Tensor):
        mask = inputs.ne(self._pad_idx).float()
        emb_layer = self.embedding(inputs)
        lstm_outputs, last_hidden = self.encoder(emb_layer, mask)

        return lstm_outputs, last_hidden


class BiLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_idx=0, num_layers=1, dropout=0.):
        super(BiLSTMDecoder, self).__init__()
        self.pad_idx = pad_idx

        self.embedding = Embedding(vocab_size, embedding_size, padding_idx=self.pad_idx)
        self.decoder_cell = BiLSTMCell(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.output_linear = Linear(hidden_size * 2, vocab_size)

    def forward(self, inputs: torch.Tensor, hidden_state: [torch.Tensor, torch.Tensor]):
        emb_layer = self.embedding(inputs)
        lstm_outputs, next_hidden_state = self.decoder_cell(emb_layer, hidden_state)
        outputs = self.output_linear(lstm_outputs)

        return outputs, next_hidden_state


class BiLSTMSeq2Seq(nn.Module):
    def __init__(self, encoder: BiLSTMEncoder, decoder: BiLSTMDecoder):
        super(BiLSTMSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.ce_loss = CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor):
        pass

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        target_seq_len = targets.size(1)
        target_mask = targets.ne(self.decoder.pad_idx).float()

        encoder_outputs, encoder_hidden = self.encoder(inputs)

        decoder_hidden = encoder_outputs
        loss = list()

        for i in range(target_seq_len):
            decoder_output, next_decoder_hidden = self.decoder(targets[:, i], decoder_hidden)
            step_loss = self.ce_loss(decoder_output, targets.squeeze(-1))

            loss.append(step_loss)
            decoder_hidden = next_decoder_hidden

        loss = torch.stack(loss, dim=1) * target_mask
        loss = torch.mean(loss, 1)

        return loss
