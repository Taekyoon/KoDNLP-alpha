from utils import load_text

from evaluator.eval_slu_model import SLUModelEvaluator
from evaluator.eval_ner_model import NERModelEvaluator
from evaluator.eval_word_segment_model import WordSegmentModelEvaluator
from evaluator.eval_translator_model import TranslationModelEvaluator


def create_evaluator(type, model, data_builder, dataset_configs, limit_len=None):
    if type == 'ner':
        eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['input'],
                                                              dataset_configs['label'])
    elif type == 'slu':
        eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['input'],
                                                              dataset_configs['slots'],
                                                              dataset_configs['intents'])
    elif type == 'translate':
        eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['source'],
                                                                  dataset_configs['target'])
    else:
        raise ValueError()

    if type == 'ner':
        evaluator = NERModelEvaluator(model,
                                      eval_data_loader)
    elif type == 'word_segment':
        dataset = load_text(dataset_configs['input'])
        input_vocab = data_builder.source_vocab
        tag_vocab = data_builder.target_vocab

        evaluator = WordSegmentModelEvaluator(model, dataset, input_vocab, tag_vocab, limit_len=limit_len)
    elif type == 'slu':
        evaluator = SLUModelEvaluator(model,
                                      eval_data_loader)
    elif type == 'translate':
        evaluator = TranslationModelEvaluator(model,
                                              eval_data_loader)
    else:
        raise ValueError()

    return evaluator
