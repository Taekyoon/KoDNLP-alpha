from pathlib import Path

from utils import parse_args, load_json, load_model

from data.builder import NERDatasetBuilder, SLUDatasetBuilder
from model.bilstm_crf import BiLSTM_CRF, BiLSTM_CRF_SLU
from train.train_ner_model import NERModelTrainer
from train.train_slu_model import SLUModelTrainer
from eval.eval_slu_model import SLUModelEvaluator


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
                                  deploy_path=deploy_path,
                                  gpu_device=gpu_device)
    elif type == 'slu':
        trainer = SLUModelTrainer(train_data_loader,
                                  valid_data_loader,
                                  model,
                                  train_configs['epochs'],
                                  train_configs['eval_steps'],
                                  deploy_path=deploy_path,
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


def main(configs):
    task_type = configs['type']
    train_dataset_configs = configs['dataset']['train']
    test_dataset_configs = configs['dataset']['test'] if 'test' in configs['dataset'] else None
    model_configs = configs['model']
    train_configs = configs['train']
    deploy_path = Path(configs['deploy']['path'])
    gpu_device = configs['gpu_device']

    data_builder = create_builder(task_type, train_dataset_configs, deploy_path=deploy_path / 'dataset')
    model = create_model(task_type, data_builder, model_configs)
    trainer = create_trainer(task_type, model, data_builder, train_configs,
                             gpu_device=gpu_device, deploy_path=deploy_path / 'model')

    print(model)
    trainer.train()

    if test_dataset_configs is not None:
        best_model_path = deploy_path / 'model' / 'best_val.pkl'
        test_model = create_model(task_type, data_builder, model_configs)
        test_model = load_model(best_model_path, test_model)
        print(test_model)
        evaluator = create_evaluator(task_type, test_model, data_builder, test_dataset_configs)
        evaluator.eval()
        print(evaluator.summary())


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    main(configs)
