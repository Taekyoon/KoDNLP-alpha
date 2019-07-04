import torch
import logging
from pathlib import Path

from data_manager.utils import create_builder
from model.utils import create_model
from trainer.utils import create_trainer
from evaluator.utils import create_evaluator

from data_manager.utils import load_vocab_dir
from data_manager.tokenizer import SyllableTokenizer
from postpro.word_segment import segment_word_by_tags

from utils import load_model, load_json

logger = logging.getLogger(__name__)


class Agent(object):
    def _run(self, query: str):
        raise NotImplementedError()

    def __call__(self, query: str):
        return self._run(query)


class WordSegmentAgent(Agent):
    def __init__(self, configs_path):
        self.configs = load_json(configs_path)

        self.task_type = self.configs['type']
        self.tokenizer_type = self.configs['tokenizer'] if 'tokenizer' in self.configs else ''
        self.deploy_path = Path(self.configs['deploy']['path'])
        self.model_configs = self.configs['model']
        self.best_model_path = self.deploy_path / 'model' / 'best_val.pkl'

        if self.tokenizer_type == 'syllable_tokenizer':
            tokenizer = SyllableTokenizer()
        else:
            raise ValueError()

        vocabs = load_vocab_dir(self.task_type, self.deploy_path)

        if 'input_vocab' in vocabs:
            self.model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

        if 'label_vocab' in vocabs:
            tag_vocab = vocabs['label_vocab']

        model = create_model(self.task_type, tag_vocab, self.model_configs)
        model = load_model(self.best_model_path, model)
        model.eval()

        self.tokenizer = tokenizer
        self.vocab = vocabs['input_vocab']
        self.label = vocabs['label_vocab']
        self.model = model
        self.preprocess = lambda x: x
        self.postprocess = segment_word_by_tags

    def _run(self, query: str):
        prepro_query = self.preprocess(query)
        tokenized_query = self.tokenizer.tokenize(prepro_query).split()
        indiced_query = self.vocab.to_indices(tokenized_query)

        model_inputs = torch.Tensor([indiced_query]).long()

        pred_score, tag_seq = self.model(model_inputs)
        labeled_tag_seq = self.label.to_tokens(tag_seq[0].tolist())

        post_processed = self.postprocess(prepro_query, labeled_tag_seq)

        return post_processed

    def train(self):
        train_configs = self.configs['train']
        train_dataset_configs = self.configs['dataset']['train']
        gpu_device = self.configs['gpu_device']

        data_builder = create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')

        if data_builder.word_to_idx:
            self.model_configs['vocab_size'] = len(data_builder.word_to_idx)

        if data_builder.tag_to_idx:
            tag_to_idx = data_builder.tag_to_idx

        model = create_model(self.task_type, tag_to_idx, self.model_configs)
        if 'load_model' in self.configs:
            logger.info('load model: {}'.format(self.configs['load_model']))
            if 'load_model_strict' in self.configs:
                strict = self.configs['load_model_strict']
            else:
                strict = False
            logger.info('set load model as strict method: {}'.format(strict))
            model = load_model(self.configs['load_model'], model, strict=strict)
        trainer = create_trainer(self.task_type, model, data_builder, train_configs,
                                 gpu_device=gpu_device, deploy_path=self.deploy_path / 'model')

        logger.info(model)
        trainer.train()

    def eval(self):
        test_dataset_configs = self.configs['dataset']['test'] if 'test' in self.configs['dataset'] else None
        train_dataset_configs = self.configs['dataset']['train']

        limit_len = test_dataset_configs['limit_len'] if 'limit_len' in test_dataset_configs else None

        data_builder = create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')
        test_dataset_configs = self.configs['dataset']['test'] if 'test' in self.configs['dataset'] else None
        evaluator = create_evaluator(self.task_type, self.model, data_builder, test_dataset_configs,
                                     limit_len)
        evaluator.eval()
        logger.info(evaluator.summary())
