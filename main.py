import logging
from pathlib import Path

from utils import parse_args, load_json, load_model, set_logging_config
from data.utils import create_builder
from model.utils import create_crf_model
from train.utils import create_trainer
from eval.utils import create_evaluator

logger = logging.getLogger(__name__)


def main(configs):
    task_type = configs['type']
    train_dataset_configs = configs['dataset']['train']
    test_dataset_configs = configs['dataset']['test'] if 'test' in configs['dataset'] else None
    model_configs = configs['model']
    train_configs = configs['train']
    deploy_path = Path(configs['deploy']['path'])
    gpu_device = configs['gpu_device']

    set_logging_config(deploy_path)

    data_builder = create_builder(task_type, train_dataset_configs, deploy_path=deploy_path / 'dataset')

    if data_builder.word_to_idx:
        model_configs['vocab_size'] = len(data_builder.word_to_idx)

    if task_type == 'slu' and data_builder.class_to_idx:
        model_configs['class_size'] = len(data_builder.class_to_idx)

    if data_builder.tag_to_idx:
        tag_to_idx = data_builder.tag_to_idx

    model = create_crf_model(task_type, tag_to_idx, model_configs)
    trainer = create_trainer(task_type, model, data_builder, train_configs,
                             gpu_device=gpu_device, deploy_path=deploy_path / 'model')

    logger.info(model)
    trainer.train()

    if test_dataset_configs is not None:
        best_model_path = deploy_path / 'model' / 'best_val.pkl'
        logger.info('load the best model')
        test_model = create_crf_model(task_type, tag_to_idx, model_configs)
        test_model = load_model(best_model_path, test_model)
        evaluator = create_evaluator(task_type, test_model, data_builder, test_dataset_configs)
        evaluator.eval()
        logger.info(evaluator.summary())


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    main(configs)
