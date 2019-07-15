import os
import logging
from typing import List, Dict
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.constants import INPUT_VOCAB_FILENAME, TAG_VOCAB_FILENAME, CLASS_VOCAB_FILENAME, \
    TRAIN_DATASET_FILENAME, VALIDATION_DATASET_FILENAME, INSTANT_DATASET_FILENAME, RANDOM_SEED, SRC_VOCAB_FILENAME, \
    TGT_VOCAB_FILENAME

from data_manager.vocab import Vocabulary
from data_manager.dataset import SequenceTagDatasetFromJSONFile, JointClsNTagDatasetFromJSONFile, \
    SequencePairDatasetFromJSONFile

from utils import make_dir_if_not_exist, load_text

from prepro.word_segment import labelize, remove_multiple_spaces

logger = logging.getLogger(__name__)


class DatasetBuilder(object):
    @property
    def input_vocab(self):
        return self._src_vocab

    @property
    def tag_vocab(self):
        return self._tgt_vocab

    @property
    def class_vocab(self):
        return self._cls_vocab

    @property
    def word_to_idx(self):
        return self._src_vocab.word_to_idx

    @property
    def tag_to_idx(self):
        return self._tgt_vocab.word_to_idx

    @property
    def class_to_idx(self):
        return self._cls_vocab.word_to_idx

    def _split_into_valid_and_train(self, input, label, test_size=0.1, random_state=RANDOM_SEED):
        raise NotImplementedError()

    def _numerize_from_text(self, data: List[str], vocab: Vocabulary):
        splited_data = self._splitify(data)

        return [vocab.to_indices(s_d) for s_d in splited_data]

    def _splitify(self, data: List[str]) -> List[List]:
        return [s.split() for s in data]

    def _load_text(self, path: Path) -> List[str]:
        logger.info('load text dataset: {}'.format(path))
        return load_text(path)

    def _save_as_json(self, obj: Dict, json_path: str) -> None:
        with open(json_path, 'w') as jsonfile:
            json.dump(obj, jsonfile, indent=4)

        return

    def _build_dataset_dir(self):
        logger.info('build dataset directory...')
        make_dir_if_not_exist(self._dataset_dir)


class NERDatasetBuilder(DatasetBuilder):
    def __init__(self,
                 input_path: Path,
                 label_path: Path,
                 file_type: str = 'text',
                 input_vocab: Vocabulary = None,
                 label_vocab: Vocabulary = None,
                 dataset_dir: str = Path('./dataset/ner')):

        self._dataset_dir = dataset_dir
        self._has_resource = False

        if os.path.isdir(self._dataset_dir):
            try:
                train_data_save_path = self._dataset_dir / TRAIN_DATASET_FILENAME
                valid_data_save_path = self._dataset_dir / VALIDATION_DATASET_FILENAME

                input_vocab_path = self._dataset_dir / INPUT_VOCAB_FILENAME
                label_vocab_path = self._dataset_dir / TAG_VOCAB_FILENAME

                if not os.path.exists(train_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(valid_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(input_vocab_path):
                    raise FileNotFoundError()

                if not os.path.exists(label_vocab_path):
                    raise FileNotFoundError()

                self._src_vocab = Vocabulary().from_json(input_vocab_path)
                self._tgt_vocab = Vocabulary().from_json(label_vocab_path)

                self._train_data_path = [train_data_save_path]
                self._valid_data_path = [valid_data_save_path]

                self._has_resource = True

                return
            except:
                raise ValueError()

        self._input_path = input_path
        self._label_path = label_path
        self._file_type = file_type

        if file_type == 'text':
            self._raw_input = self._load_text(self._input_path)
            self._raw_label = self._load_text(self._label_path)
        else:
            raise NotImplementedError()

        self._src_vocab = input_vocab
        self._tgt_vocab = label_vocab

        self._train_data_path = list()
        self._valid_data_path = list()

        self._build_dataset_dir()

    def build_vocabulary(self,
                         max_size: int = None,
                         min_freq: int = 1) -> None:
        if self._has_resource:
            return

        input_vocab_path = self._dataset_dir / INPUT_VOCAB_FILENAME
        label_vocab_path = self._dataset_dir / TAG_VOCAB_FILENAME

        if self._src_vocab is None:
            logger.info('build input text vocabulary...')
            self._src_vocab = Vocabulary(max_size=max_size, min_freq=min_freq, bos_token=None, eos_token=None)
            input_data = self._splitify(self._raw_input)
            self._src_vocab.fit(input_data)

        if self._tgt_vocab is None:
            logger.info('build label vocabulary...')
            self._tgt_vocab = Vocabulary(unknown_token=None)
            label_data = self._splitify(self._raw_label)
            self._tgt_vocab.fit(label_data)

        self._src_vocab.to_json(input_vocab_path)
        logger.info('save input text vocabulary...')
        self._tgt_vocab.to_json(label_vocab_path)
        logger.info('save label vocabulary...')

        return

    def build_trainable_dataset(self,
                                train_data_path: str = None,
                                valid_data_path: str = None) -> None:
        if self._has_resource:
            return

        if self._src_vocab is None or self._tgt_vocab is None:
            raise ValueError()

        train_data = dict()
        train_data_path = self._dataset_dir / TRAIN_DATASET_FILENAME if train_data_path is None else train_data_path

        valid_data = dict()
        valid_data_path = self._dataset_dir / VALIDATION_DATASET_FILENAME if valid_data_path is None else valid_data_path

        logger.info('split train and valid dataset: test split rate is 0.1')
        train_raw_data, valid_raw_data = self._split_into_valid_and_train(self._raw_input, self._raw_label)

        train_data['inputs'] = self._numerize_from_text(train_raw_data[0], self._src_vocab)
        train_data['entities'] = self._numerize_from_text(train_raw_data[1], self._tgt_vocab)

        valid_data['inputs'] = self._numerize_from_text(valid_raw_data[0], self._src_vocab)
        valid_data['entities'] = self._numerize_from_text(valid_raw_data[1], self._tgt_vocab)

        logger.info('save train and valid dataset as json format.')
        self._save_as_json(train_data, train_data_path)
        self._save_as_json(valid_data, valid_data_path)

        self._train_data_path.append(train_data_path)
        self._valid_data_path.append(valid_data_path)

        return

    def build_instant_data_loader(self, input_path, label_path, data_path=None):
        instant_data = dict()
        data_path = self._dataset_dir / INSTANT_DATASET_FILENAME if data_path is None else data_path

        input_data = self._load_text(input_path)
        label_data = self._load_text(label_path)

        instant_data['inputs'] = self._numerize_from_text(input_data, self._src_vocab)
        instant_data['entities'] = self._numerize_from_text(label_data, self._tgt_vocab)

        self._save_as_json(instant_data, data_path)

        instant_dataset = SequenceTagDatasetFromJSONFile(data_path)

        instant_data_loader = DataLoader(instant_dataset,
                                         batch_size=1)

        return instant_data_loader

    def build_data_loader(self, batch_size, limit_pad_len, valid_batch_size=1, enable_length=True):
        logger.info('now get training dataloader object...')
        train_dataset = SequenceTagDatasetFromJSONFile(self._train_data_path[0],
                                                       limit_pad_len=limit_pad_len,
                                                       enable_length=enable_length)

        if valid_batch_size <= 1:
            limit_pad_len = None

        valid_dataset = SequenceTagDatasetFromJSONFile(self._valid_data_path[0],
                                                       limit_pad_len=limit_pad_len)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=True)

        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=valid_batch_size,
                                       num_workers=4)

        return train_data_loader, valid_data_loader

    def _split_into_valid_and_train(self, input, label, test_size=0.1, random_state=RANDOM_SEED):
        input_train, input_test, label_train, label_test = train_test_split(input, label,
                                                                            test_size=test_size,
                                                                            random_state=random_state)

        return (input_train, label_train), (input_test, label_test)


class SLUDatasetBuilder(DatasetBuilder):
    def __init__(self,
                 input_path: Path,
                 label_path: Path,
                 class_path: Path,
                 file_type: str = 'text',
                 input_vocab: Vocabulary = None,
                 label_vocab: Vocabulary = None,
                 class_vocab: Vocabulary = None,
                 dataset_dir: str = Path('./dataset/slu')):
        self._dataset_dir = dataset_dir
        self._has_resource = False

        if os.path.isdir(self._dataset_dir):
            try:
                train_data_save_path = self._dataset_dir / TRAIN_DATASET_FILENAME
                valid_data_save_path = self._dataset_dir / VALIDATION_DATASET_FILENAME

                input_vocab_path = self._dataset_dir / INPUT_VOCAB_FILENAME
                label_vocab_path = self._dataset_dir / TAG_VOCAB_FILENAME
                class_vocab_path = self._dataset_dir / CLASS_VOCAB_FILENAME

                if not os.path.exists(train_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(valid_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(input_vocab_path):
                    raise FileNotFoundError()

                if not os.path.exists(label_vocab_path):
                    raise FileNotFoundError()

                if not os.path.exists(class_vocab_path):
                    raise FileNotFoundError()

                self._src_vocab = Vocabulary().from_json(input_vocab_path)
                self._tgt_vocab = Vocabulary().from_json(label_vocab_path)
                self._cls_vocab = Vocabulary().from_json(class_vocab_path)

                self._train_data_path = [train_data_save_path]
                self._valid_data_path = [valid_data_save_path]

                self._has_resource = True

                return
            except:
                raise ValueError()

        self._input_path = input_path
        self._label_path = label_path
        self._class_path = class_path
        self._file_type = file_type

        if file_type == 'text':
            self._raw_input = self._load_text(self._input_path)
            self._raw_label = self._load_text(self._label_path)
            self._raw_class = self._load_text(self._class_path)
        else:
            raise NotImplementedError()

        self._src_vocab = input_vocab
        self._tgt_vocab = label_vocab
        self._cls_vocab = class_vocab

        self._train_data_path = list()
        self._valid_data_path = list()

        self._build_dataset_dir()

    def build_vocabulary(self,
                         max_size: int = None,
                         min_freq: int = 1) -> None:
        if self._has_resource:
            return

        input_vocab_path = self._dataset_dir / INPUT_VOCAB_FILENAME
        label_vocab_path = self._dataset_dir / TAG_VOCAB_FILENAME
        class_vocab_path = self._dataset_dir / CLASS_VOCAB_FILENAME

        if self._src_vocab is None:
            logger.info('build input text vocabulary...')
            self._src_vocab = Vocabulary(max_size=max_size, min_freq=min_freq, bos_token=None, eos_token=None)
            input_data = self._splitify(self._raw_input)
            self._src_vocab.fit(input_data)

        if self._tgt_vocab is None:
            logger.info('build label vocabulary...')
            self._tgt_vocab = Vocabulary(unknown_token=None)
            label_data = self._splitify(self._raw_label)
            self._tgt_vocab.fit(label_data)

        if self._cls_vocab is None:
            logger.info('build class vocabulary...')
            self._cls_vocab = Vocabulary(unknown_token=None, padding_token=None, bos_token=None, eos_token=None)
            class_data = self._raw_class
            self._cls_vocab.fit(class_data)

        logger.info('save input text vocabulary...')
        self._src_vocab.to_json(input_vocab_path)
        logger.info('save label vocabulary...')
        self._tgt_vocab.to_json(label_vocab_path)
        logger.info('save class vocabulary...')
        self._cls_vocab.to_json(class_vocab_path)

        return

    def build_trainable_dataset(self,
                                train_data_save_path: str = None,
                                valid_data_save_path: str = None) -> None:
        if self._has_resource:
            return

        if self._src_vocab is None or self._tgt_vocab is None:
            raise ValueError()

        train_data = dict()
        train_data_save_path = self._dataset_dir / TRAIN_DATASET_FILENAME if train_data_save_path is None else train_data_save_path

        valid_data = dict()
        valid_data_save_path = self._dataset_dir / VALIDATION_DATASET_FILENAME if valid_data_save_path is None else valid_data_save_path

        logger.info('split train and valid dataset: test split rate is 0.1')
        train_raw_data, valid_raw_data = self._split_into_valid_and_train(self._raw_input, self._raw_label,
                                                                          self._raw_class)

        train_data['inputs'] = self._numerize_from_text(train_raw_data[0], self._src_vocab)
        train_data['slots'] = self._numerize_from_text(train_raw_data[1], self._tgt_vocab)
        train_data['intents'] = self._numerize_from_text(train_raw_data[2], self._cls_vocab)

        valid_data['inputs'] = self._numerize_from_text(valid_raw_data[0], self._src_vocab)
        valid_data['slots'] = self._numerize_from_text(valid_raw_data[1], self._tgt_vocab)
        valid_data['intents'] = self._numerize_from_text(valid_raw_data[2], self._cls_vocab)

        logger.info('save train and valid dataset as json format.')
        self._save_as_json(train_data, train_data_save_path)
        self._save_as_json(valid_data, valid_data_save_path)

        self._train_data_path.append(train_data_save_path)
        self._valid_data_path.append(valid_data_save_path)

        return

    def build_instant_data_loader(self, input_path, label_path, class_path, data_path=None):
        instant_data = dict()
        data_path = self._dataset_dir / INSTANT_DATASET_FILENAME if data_path is None else data_path

        input_data = self._load_text(input_path)
        label_data = self._load_text(label_path)
        class_data = self._load_text(class_path)

        instant_data['inputs'] = self._numerize_from_text(input_data, self._src_vocab)
        instant_data['slots'] = self._numerize_from_text(label_data, self._tgt_vocab)
        instant_data['intents'] = self._numerize_from_text(class_data, self._cls_vocab)

        self._save_as_json(instant_data, data_path)

        instant_dataset = JointClsNTagDatasetFromJSONFile(data_path)

        instant_data_loader = DataLoader(instant_dataset,
                                         batch_size=1)

        return instant_data_loader

    def build_data_loader(self, batch_size, limit_pad_len, valid_batch_size=1, enable_length=True):
        logger.info('now get training dataloader object...')
        train_dataset = JointClsNTagDatasetFromJSONFile(self._train_data_path[0],
                                                        limit_pad_len=limit_pad_len,
                                                        enable_length=enable_length)
        if valid_batch_size <= 1:
            limit_pad_len = None

        valid_dataset = JointClsNTagDatasetFromJSONFile(self._valid_data_path[0],
                                                        limit_pad_len=limit_pad_len)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=True)

        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=valid_batch_size)

        return train_data_loader, valid_data_loader

    def _split_into_valid_and_train(self, input, label, cls, test_size=0.1, random_state=RANDOM_SEED):
        input_train, input_test, label_train, label_test, class_train, class_test = train_test_split(input, label, cls,
                                                                                                     test_size=test_size,
                                                                                                     random_state=random_state)

        return (input_train, label_train, class_train), (input_test, label_test, class_test)


class WordSegmentationDatasetBuilder(NERDatasetBuilder):
    def __init__(self,
                 input_path: Path,
                 file_type: str = 'text',
                 input_vocab: Vocabulary = None,
                 label_vocab: Vocabulary = None,
                 bi_tags_only: bool = False,
                 dataset_dir: str = Path('./dataset/word_segment')):

        self._dataset_dir = dataset_dir
        self._has_resource = False

        if input_vocab is not None:
            logger.info('use existing input vocabulary.')

        if os.path.isdir(self._dataset_dir):
            try:
                train_data_save_path = self._dataset_dir / TRAIN_DATASET_FILENAME
                valid_data_save_path = self._dataset_dir / VALIDATION_DATASET_FILENAME

                input_vocab_path = self._dataset_dir / INPUT_VOCAB_FILENAME
                label_vocab_path = self._dataset_dir / TAG_VOCAB_FILENAME

                if not os.path.exists(train_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(valid_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(input_vocab_path):
                    raise FileNotFoundError()

                if not os.path.exists(label_vocab_path):
                    raise FileNotFoundError()

                self._src_vocab = Vocabulary().from_json(input_vocab_path)
                self._tgt_vocab = Vocabulary().from_json(label_vocab_path)

                self._train_data_path = [train_data_save_path]
                self._valid_data_path = [valid_data_save_path]

                self._has_resource = True

                return
            except:
                raise ValueError()

        self._bi_tags_only = bi_tags_only
        self._input_path = input_path
        self._file_type = file_type

        self._label = None

        if file_type == 'text':
            self._input_text = self._load_text(self._input_path)
        else:
            raise NotImplementedError()

        logger.info('now labelize dataset...')
        self._raw_input, self._raw_label = self._self_labelize(self._input_text)

        self._src_vocab = input_vocab
        self._tgt_vocab = label_vocab

        self._train_data_path = list()
        self._valid_data_path = list()

        self._build_dataset_dir()

    def _self_labelize(self, text_dataset):
        inputs, labels = list(), list()

        for s in text_dataset:
            s = remove_multiple_spaces(s)
            s, t = labelize(s, bi_tags_only=self._bi_tags_only)

            s = ' '.join([ch for ch in s])
            t = ' '.join([ch for ch in t])

            inputs.append(s)
            labels.append(t)

        return inputs, labels


class SequencePairDatasetBuilder(DatasetBuilder):
    def __init__(self,
                 src_path: Path,
                 tgt_path: Path,
                 file_type: str = 'text',
                 src_vocab: Vocabulary = None,
                 tgt_vocab: Vocabulary = None,
                 dataset_dir: Path = Path('./dataset/seq_pair')):

        self._dataset_dir = dataset_dir
        self._has_resource = False

        if self._dataset_dir.exists():
            try:
                train_data_save_path = self._dataset_dir / TRAIN_DATASET_FILENAME
                valid_data_save_path = self._dataset_dir / VALIDATION_DATASET_FILENAME

                src_vocab_path = self._dataset_dir / SRC_VOCAB_FILENAME
                tgt_vocab_path = self._dataset_dir / TGT_VOCAB_FILENAME

                if not os.path.exists(train_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(valid_data_save_path):
                    raise FileNotFoundError()

                if not os.path.exists(src_vocab_path):
                    raise FileNotFoundError()

                if not os.path.exists(tgt_vocab_path):
                    raise FileNotFoundError()

                self._src_vocab = Vocabulary().from_json(src_vocab_path)
                self._tgt_vocab = Vocabulary().from_json(tgt_vocab_path)

                self._train_data_path = [train_data_save_path]
                self._valid_data_path = [valid_data_save_path]

                self._has_resource = True

                return
            except:
                raise ValueError()

        self._src_path = src_path
        self._tgt_path = tgt_path
        self._file_type = file_type

        if file_type == 'text':
            self._raw_src = self._load_text(self._src_path)
            self._raw_tgt = self._load_text(self._tgt_path)
        else:
            raise NotImplementedError()

        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

        self._train_data_path = list()
        self._valid_data_path = list()

        self._build_dataset_dir()

    def build_vocabulary(self,
                         max_size: int = None,
                         min_freq: int = 1) -> None:
        if self._has_resource:
            return

        src_vocab_path = self._dataset_dir / SRC_VOCAB_FILENAME
        tgt_vocab_path = self._dataset_dir / TGT_VOCAB_FILENAME

        if self._src_vocab is None:
            logger.info('build source text vocabulary...')
            self._src_vocab = Vocabulary(max_size=max_size, min_freq=min_freq, bos_token=None, eos_token=None)
            src_data = self._splitify(self._raw_src)
            self._src_vocab.fit(src_data)

        if self._tgt_vocab is None:
            logger.info('build target text vocabulary...')
            self._tgt_vocab = Vocabulary(max_size=max_size, min_freq=min_freq, bos_token=None, eos_token=None)
            tgt_data = self._splitify(self._raw_tgt)
            self._tgt_vocab.fit(tgt_data)

        self._src_vocab.to_json(src_vocab_path)
        logger.info('save input text vocabulary...')
        self._tgt_vocab.to_json(tgt_vocab_path)
        logger.info('save label vocabulary...')

        return

    def build_trainable_dataset(self,
                                train_data_path: str = None,
                                valid_data_path: str = None) -> None:
        if self._has_resource:
            return

        if self._src_vocab is None or self._tgt_vocab is None:
            raise ValueError()

        train_data = dict()
        train_data_path = self._dataset_dir / TRAIN_DATASET_FILENAME if train_data_path is None else train_data_path

        valid_data = dict()
        valid_data_path = self._dataset_dir / VALIDATION_DATASET_FILENAME if valid_data_path is None else valid_data_path

        logger.info('split train and valid dataset: test split rate is 0.1')
        train_raw_data, valid_raw_data = self._split_into_valid_and_train(self._raw_src, self._raw_tgt)

        train_data['sources'] = self._numerize_from_text(train_raw_data[0], self._src_vocab)
        train_data['targets'] = self._numerize_from_text(train_raw_data[1], self._tgt_vocab)

        valid_data['sources'] = self._numerize_from_text(valid_raw_data[0], self._src_vocab)
        valid_data['targets'] = self._numerize_from_text(valid_raw_data[1], self._tgt_vocab)

        logger.info('save train and valid dataset as json format.')
        self._save_as_json(train_data, train_data_path)
        self._save_as_json(valid_data, valid_data_path)

        self._train_data_path.append(train_data_path)
        self._valid_data_path.append(valid_data_path)

        return

    def build_instant_data_loader(self, src_path, tgt_path, data_path=None):
        instant_data = dict()
        data_path = self._dataset_dir / INSTANT_DATASET_FILENAME if data_path is None else data_path

        src_data = self._load_text(src_path)
        tgt_data = self._load_text(tgt_path)

        instant_data['sources'] = self._numerize_from_text(src_data, self._src_vocab)
        instant_data['targets'] = self._numerize_from_text(tgt_data, self._tgt_vocab)

        self._save_as_json(instant_data, data_path)

        instant_dataset = SequencePairDatasetFromJSONFile(data_path)

        instant_data_loader = DataLoader(instant_dataset,
                                         batch_size=1)

        return instant_data_loader

    def build_data_loader(self, batch_size, limit_src_pad_len, limit_tgt_pad_len, valid_batch_size=1,
                          enable_length=True):
        logger.info('now get training dataloader object...')
        train_dataset = SequencePairDatasetFromJSONFile(self._train_data_path[0],
                                                        limit_src_pad_len=limit_src_pad_len,
                                                        limit_tgt_pad_len=limit_tgt_pad_len,
                                                        enable_length=enable_length)

        if valid_batch_size <= 1:
            limit_src_pad_len = None
            limit_tgt_pad_len = None

        valid_dataset = SequencePairDatasetFromJSONFile(self._valid_data_path[0],
                                                        limit_src_pad_len=limit_src_pad_len,
                                                        limit_tgt_pad_len=limit_tgt_pad_len)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=True)

        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=valid_batch_size,
                                       num_workers=4)

        return train_data_loader, valid_data_loader

    def _split_into_valid_and_train(self, input, label, test_size=0.1, random_state=RANDOM_SEED):
        input_train, input_test, label_train, label_test = train_test_split(input, label,
                                                                            test_size=test_size,
                                                                            random_state=random_state)

        return (input_train, label_train), (input_test, label_test)
