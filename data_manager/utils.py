from pathlib import Path

from configs.constants import INPUT_VOCAB_FILENAME, TAG_VOCAB_FILENAME, CLASS_VOCAB_FILENAME, SRC_VOCAB_FILENAME, \
                              TGT_VOCAB_FILENAME
from data_manager.builder import NERDatasetBuilder, SLUDatasetBuilder, WordSegmentationDatasetBuilder, \
    SequencePairDatasetBuilder
from data_manager.vocab import Vocabulary


def create_builder(type, dataset_configs, deploy_path='./tmp'):
    if 'vocab_path' in dataset_configs:
        input_vocab = load_vocab_file(dataset_configs['vocab_path'])
    else:
        input_vocab = None

    if type == 'ner':
        builder = NERDatasetBuilder(Path(dataset_configs['input']), Path(dataset_configs['label']),
                                    dataset_dir=deploy_path, input_vocab=input_vocab)
    elif type == 'word_segment':
        if 'bi_tags_only' not in dataset_configs:
            dataset_configs['bi_tags_only'] = False

        builder = WordSegmentationDatasetBuilder(Path(dataset_configs['input']), dataset_dir=deploy_path,
                                                 input_vocab=input_vocab, bi_tags_only=dataset_configs['bi_tags_only'])
    elif type == 'slu':
        builder = SLUDatasetBuilder(Path(dataset_configs['input']), Path(dataset_configs['slots']),
                                    Path(dataset_configs['intents']), dataset_dir=deploy_path,
                                    input_vocab=input_vocab)
    elif type == 'translate':
        builder = SequencePairDatasetBuilder(Path(dataset_configs['source']), Path(dataset_configs['target']),
                                             dataset_dir=deploy_path)
    else:
        raise NotImplementedError()

    if 'vocab_min_freq' in dataset_configs:
        builder.build_vocabulary(min_freq=dataset_configs['vocab_min_freq'])
    else:
        builder.build_vocabulary()

    builder.build_trainable_dataset()

    return builder


def load_vocab_file(file_path):
    return Vocabulary().from_json(file_path)


def load_vocab_dir(type, dir_path):
    if type == 'ner' or type == 'word_segment':
        input_vocab_path = dir_path / 'dataset' / INPUT_VOCAB_FILENAME
        label_vocab_path = dir_path / 'dataset' / TAG_VOCAB_FILENAME

        input_vocab = Vocabulary()
        label_vocab = Vocabulary()

        input_vocab.from_json(input_vocab_path)
        label_vocab.from_json(label_vocab_path)

        ret = {'input_vocab': input_vocab,
               'label_vocab': label_vocab}

    elif type == 'slu':
        input_vocab_path = dir_path / 'dataset' / INPUT_VOCAB_FILENAME
        label_vocab_path = dir_path / 'dataset' / TAG_VOCAB_FILENAME
        class_vocab_path = dir_path / 'dataset' / CLASS_VOCAB_FILENAME

        input_vocab = Vocabulary()
        label_vocab = Vocabulary()
        class_vocab = Vocabulary()

        input_vocab.from_json(input_vocab_path)
        label_vocab.from_json(label_vocab_path)
        class_vocab.from_json(class_vocab_path)

        ret = {'input_vocab': input_vocab,
               'label_vocab': label_vocab,
               'class_vocab': class_vocab}

    elif type == 'translate':
        source_vocab_path = dir_path / 'dataset' / SRC_VOCAB_FILENAME
        target_vocab_path = dir_path / 'dataset' / TGT_VOCAB_FILENAME

        source_vocab = Vocabulary()
        target_vocab = Vocabulary()

        source_vocab.from_json(source_vocab_path)
        target_vocab.from_json(target_vocab_path)

        ret = {'source_vocab': source_vocab,
               'target_vocab': target_vocab}

    else:
        raise ValueError()

    return ret
