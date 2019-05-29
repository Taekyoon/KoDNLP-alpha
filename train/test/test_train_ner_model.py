from pathlib import Path
from data.builder import NERDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF
from train.train_ner_model import NERModelTrainer


def test_model_train():
    embedding_dim = 100
    hidden_dim = 150
    epochs = 10
    eval_steps = 5

    input_path = './data/test/test_dataset/ner/input.txt'
    label_path = './data/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = NERDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, valid_data_loader = ner_builder.build_data_loader(batch_size, sequence_length)

    word_to_idx, tag_to_idx = ner_builder.word_to_idx, ner_builder.tag_to_idx

    model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, embedding_dim, hidden_dim)

    ner_trainer = NERModelTrainer(train_data_loader,
                                  valid_data_loader,
                                  model,
                                  epochs,
                                  eval_steps)

    ner_trainer.train()

    assert False
