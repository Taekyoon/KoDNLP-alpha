from pathlib import Path
from data.builder import SLUDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF_SLU
from train.train_slu_model import SLUModelTrainer


def test_model_train_with_train_data_over_85_f1_score():
    embedding_dim = 50
    hidden_dim = 50
    epochs = 350
    eval_steps = 10

    input_path = './data/test/test_dataset/slu/input.txt'
    label_path = './data/test/test_dataset/slu/output.txt'
    class_path = './data/test/test_dataset/slu/class.txt'

    dataset_dir = Path('./data/test/test_dataset/slu/train_dataset')

    batch_size = 2
    sequence_length = 15

    slu_builder = SLUDatasetBuilder(input_path, label_path, class_path, dataset_dir=dataset_dir)

    slu_builder.build_vocabulary()
    slu_builder.build_trainable_dataset()
    train_data_loader, _ = slu_builder.build_data_loader(batch_size, sequence_length)

    word_to_idx, tag_to_idx, class_to_idx = slu_builder.word_to_idx, slu_builder.tag_to_idx, slu_builder.class_to_idx

    slu_model = BiLSTM_CRF_SLU(len(word_to_idx), tag_to_idx, class_to_idx, embedding_dim, hidden_dim)

    slu_trainer = SLUModelTrainer(train_data_loader,
                                  train_data_loader,
                                  slu_model,
                                  epochs,
                                  eval_steps)

    slu_trainer.train()

    assert slu_trainer.best_class_val_f1_score > 0.85
    assert slu_trainer.best_tag_val_f1_score > 0.85

