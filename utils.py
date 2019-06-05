import json
import argparse
from pathlib import Path

from data.builder import NERDatasetBuilder, SLUDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF, BiLSTM_CRF_SLU
from train.train_ner_model import NERModelTrainer
from train.train_slu_model import SLUModelTrainer
from eval.eval_slu_model import SLUModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--configs_path', dest='configs_path', type=str,
                        help='configureations file path')
    args = parser.parse_args()

    return args


def load_json(path):
    with open(path) as json_file:
        json_obj = json.load(json_file)

    return json_obj


def create_builder(type, dataset_configs, deploy_path='./tmp'):
    if type == 'ner':
        builder = NERDatasetBuilder(Path(dataset_configs['input']), Path(dataset_configs['label']),
                                    dataset_dir=deploy_path)
    elif type == 'slu':
        builder = SLUDatasetBuilder(Path(dataset_configs['input']), Path(dataset_configs['slots']),
                                    Path(dataset_configs['intents']), dataset_dir=deploy_path)
    else:
        raise NotImplementedError()

    if 'vocab_min_freq' in dataset_configs:
        builder.build_vocabulary(min_freq=dataset_configs['vocab_min_freq'])
    else:
        builder.build_vocabulary()

    builder.build_trainable_dataset()

    return builder


def create_model(type, data_builder, model_configs):
    model_type = model_configs['type']
    model_params = model_configs['parameters']
    if type == 'ner':
        word_to_idx, tag_to_idx = data_builder.word_to_idx, data_builder.tag_to_idx
        if model_type == 'bilstm_crf':
            model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, model_params['word_embedding_dims'],
                               model_params['hidden_dims'])
        else:
            raise ValueError()
    elif type == 'slu':
        word_to_idx, tag_to_idx, class_to_idx = data_builder.word_to_idx, data_builder.tag_to_idx, \
                                                data_builder.class_to_idx
        num_layers = model_params['lstm_num_layers'] if 'lstm_num_layers' in model_params else 1
        dropout = model_params['lstm_dropout'] if 'lstm_dropout' in model_params else 0.5

        if model_type == 'bilstm_crf_slu':
            model = BiLSTM_CRF_SLU(len(word_to_idx), tag_to_idx, class_to_idx, model_params['word_embedding_dims'], \
                                   model_params['hidden_dims'], num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return model


def create_trainer(type, model, data_builder, train_configs, gpu_device=-1, deploy_path='./tmp'):
    train_data_loader, valid_data_loader = data_builder.build_data_loader(train_configs['batch_size'],
                                                                          train_configs['sequence_length'])

    if type == 'ner':
        trainer = NERModelTrainer(train_data_loader,
                                  valid_data_loader,
                                  model,
                                  train_configs['epochs'],
                                  train_configs['eval_steps'],
                                  gpu_device=gpu_device)
    elif type == 'slu':
        trainer = SLUModelTrainer(train_data_loader,
                                  valid_data_loader,
                                  model,
                                  train_configs['epochs'],
                                  train_configs['eval_steps'],
                                  gpu_device=gpu_device)
    else:
        raise ValueError()

    return trainer


def create_evaluator(type, model, data_builder, dataset_configs):
    eval_data_loader = data_builder.build_instant_data_loader(dataset_configs['input'],
                                                              dataset_configs['slots'],
                                                              dataset_configs['intents'])

    if type == 'ner':
        raise NotImplementedError()
    elif type == 'slu':
        evaluator = SLUModelEvaluator(model,
                                      eval_data_loader)
    else:
        raise ValueError()

    return evaluator
