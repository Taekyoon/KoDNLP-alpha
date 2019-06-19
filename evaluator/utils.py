from evaluator.eval_slu_model import SLUModelEvaluator
from evaluator.eval_ner_model import NERModelEvaluator


def create_evaluator(type, model, data_builder, dataset_configs):
    if type == 'ner':
        eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['input'],
                                                              dataset_configs['label'])
    elif type == 'slu':
        eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['input'],
                                                              dataset_configs['slots'],
                                                              dataset_configs['intents'])

    if type == 'ner':
        evaluator = NERModelEvaluator(model,
                                      eval_data_loader)
    elif type == 'slu':
        evaluator = SLUModelEvaluator(model,
                                      eval_data_loader)
    else:
        raise ValueError()

    return evaluator
