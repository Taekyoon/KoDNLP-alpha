from pathlib import Path
from data.builder import NERDatasetBuilder
from torch.utils.data import DataLoader


def test_ner_dataset_builder_build_dataloader_as_default():
    input_path = './data/test/test_dataset/ner/input.txt'
    label_path = './data/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = NERDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, valid_data_loader = ner_builder.build_data_loader(batch_size, sequence_length)

    assert isinstance(train_data_loader, DataLoader)
    assert isinstance(valid_data_loader, DataLoader)


def test_ner_dataset_lodaer_iterate():
    input_path = './data/test/test_dataset/ner/input.txt'
    label_path = './data/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = NERDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, valid_data_loader = ner_builder.build_data_loader(batch_size, sequence_length)

    for batch in train_data_loader:
        train_batch = batch
        break

    assert isinstance(train_batch, dict)
    assert len(train_batch['inputs']) == batch_size
    assert train_batch['inputs']['length'][0] <= sequence_length
