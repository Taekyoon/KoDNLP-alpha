from pathlib import Path
from data.builder import SLUDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF_SLU
from train.train_slu_model import SLUModelTrainer


def test_model_train_with_train_data():
    embedding_dim = 50
    hidden_dim = 50
    epochs = 400
    eval_steps = 10

    input_path = './train/test/test_dataset/slu/input.txt'
    label_path = './train/test/test_dataset/slu/output.txt'
    class_path = './train/test/test_dataset/slu/class.txt'

    deploy_dir = Path('./train/test/test_dataset/slu/train_dataset')

    batch_size = 2
    sequence_length = 15

    slu_builder = SLUDatasetBuilder(input_path, label_path, class_path, dataset_dir=deploy_dir / 'dataset')

    slu_builder.build_vocabulary()
    slu_builder.build_trainable_dataset()
    train_data_loader, valid_data_loader = slu_builder.build_data_loader(batch_size, sequence_length)

    word_to_idx, tag_to_idx, class_to_idx = slu_builder.word_to_idx, slu_builder.tag_to_idx, slu_builder.class_to_idx

    slu_model = BiLSTM_CRF_SLU(len(word_to_idx), tag_to_idx, class_to_idx, embedding_dim, hidden_dim)

    slu_trainer = SLUModelTrainer(train_data_loader,
                                  valid_data_loader,
                                  slu_model,
                                  epochs,
                                  eval_steps,
                                  deploy_path=deploy_dir / 'model')

    slu_trainer.train()

    assert slu_trainer.train_loss < 15

