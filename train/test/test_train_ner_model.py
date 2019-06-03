from pathlib import Path
from data.builder import NERDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF
from train.train_ner_model import NERModelTrainer


def test_model_train_with_train_data_over_90_f1_score():
    embedding_dim = 50
    hidden_dim = 50
    epochs = 400
    eval_steps = 10

    input_path = './train/test/test_dataset/ner/input.txt'
    label_path = './train/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./train/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 15

    ner_builder = NERDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, _ = ner_builder.build_data_loader(batch_size, sequence_length)

    word_to_idx, tag_to_idx = ner_builder.word_to_idx, ner_builder.tag_to_idx

    ner_model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, embedding_dim, hidden_dim)

    ner_trainer = NERModelTrainer(train_data_loader,
                                  train_data_loader,
                                  ner_model,
                                  epochs,
                                  eval_steps)

    ner_trainer.train()

    assert ner_trainer.best_val_f1_score > 0.90
