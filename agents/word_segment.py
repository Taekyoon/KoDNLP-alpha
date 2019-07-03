import torch

from pathlib import Path

from model.utils import create_model

from data_manager.utils import load_vocab_dir
from data_manager.tokenizer import SyllableTokenizer
from postpro.word_segment import segment_word_by_tags

from utils import load_model, load_json


class Agent(object):
    def _run(self, query: str):
        raise NotImplementedError()

    def __call__(self, query: str):
        return self._run(query)


class WordSegmentAgent(Agent):
    def __init__(self, configs_path):
        configs = load_json(configs_path)

        task_type = configs['type']
        tokenizer_type = configs['tokenizer'] if 'tokenizer' in configs else ''
        deploy_path = Path(configs['deploy']['path'])
        model_configs = configs['model']
        best_model_path = deploy_path / 'model' / 'best_val.pkl'

        if tokenizer_type == 'syllable_tokenizer':
            tokenizer = SyllableTokenizer()
        else:
            raise ValueError()

        vocabs = load_vocab_dir(task_type, deploy_path)

        if 'input_vocab' in vocabs:
            model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

        if 'label_vocab' in vocabs:
            tag_vocab = vocabs['label_vocab']

        model = create_model(task_type, tag_vocab, model_configs)
        model = load_model(best_model_path, model)
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
