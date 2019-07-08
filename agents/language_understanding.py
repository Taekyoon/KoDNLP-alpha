import os
import torch
import logging
from pathlib import Path

from agents import Agent

from data_manager.utils import create_builder
from model.utils import create_model
from trainer.utils import create_trainer
from evaluator.utils import create_evaluator

from data_manager.utils import load_vocab_dir
from data_manager.tokenizer import SyllableTokenizer
from postpro.ner import process_by_ner

from utils import load_model, load_json, register_logging

logger = logging.getLogger(__name__)


@register_logging
class SLUAgent(Agent):
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

        if not os.path.exists(self.deploy_path / 'dataset'):
            train_dataset_configs = self.configs['dataset']['train']
            create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')

        vocabs = load_vocab_dir(self.task_type, self.deploy_path)

        if 'input_vocab' in vocabs:
            self.model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

        if 'class_vocab' in vocabs:
            self.model_configs['class_size'] = len(vocabs['class_vocab'].word_to_idx)

        if 'label_vocab' in vocabs:
            tag_vocab = vocabs['label_vocab']

        model = create_model(self.task_type, tag_vocab, self.model_configs)
        if os.path.exists(self.best_model_path):
            model = load_model(self.best_model_path, model)
        model.eval()

        self.tokenizer = tokenizer
        self.vocab = vocabs['input_vocab']
        self.label = vocabs['label_vocab']
        self.intent = vocabs['class_vocab']
        self.model = model
        self.preprocess = lambda x: x
        self.postprocess = process_by_ner

    def _run(self, query: str):
        prepro_query = self.preprocess(query)
        tokenized_query = self.tokenizer.tokenize(prepro_query).split()
        indiced_query = self.vocab.to_indices(tokenized_query)

        model_inputs = torch.Tensor([indiced_query]).long()

        pred_score, tag_seq, intent_probs = self.model(model_inputs)
        labeled_tag_seq = self.label.to_tokens(tag_seq[0].tolist())

        intent = self.intent.to_tokens(torch.argmax(intent_probs, dim=-1).tolist())[0]
        most_intent_prob = intent_probs.tolist()[0][torch.argmax(intent_probs, dim=-1).tolist()[0]]

        slot_tags, slots = self.postprocess(prepro_query, labeled_tag_seq)
        entities = [{'entity': tag, 'value': entity.replace(' ', '').replace('_', ' ').strip()} for tag, entity in
                    zip(slot_tags, slots)]

        outputs = {'input': prepro_query,
                   'label': labeled_tag_seq,
                   'sequence_score': float(pred_score.detach().numpy()),
                   'intent': {'name': intent, 'prob': most_intent_prob},
                   'slots': entities}

        return outputs

    def train(self):
        train_configs = self.configs['train']
        train_dataset_configs = self.configs['dataset']['train']
        gpu_device = self.configs['gpu_device']

        data_builder = create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')

        if data_builder.word_to_idx:
            self.model_configs['vocab_size'] = len(data_builder.word_to_idx)

        if data_builder.class_to_idx:
            self.model_configs['class_size'] = len(data_builder.class_to_idx)

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
