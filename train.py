import logging
from pathlib import Path

from utils import parse_args, load_json, load_model, set_logging_config
from data_manager.utils import create_builder
from model.utils import create_model
from trainer.utils import create_trainer
from evaluator.utils import create_evaluator

logger = logging.getLogger(__name__)


def main(configs, test_only):
    task_type = configs['type']
    train_dataset_configs = configs['dataset']['train']
    test_dataset_configs = configs['dataset']['test'] if 'test' in configs['dataset'] else None
    model_configs = configs['model']
    train_configs = configs['train']
    deploy_path = Path(configs['deploy']['path'])
    gpu_device = configs['gpu_device']

    set_logging_config(deploy_path)

    data_builder = create_builder(task_type, train_dataset_configs, deploy_path=deploy_path / 'dataset')

    if data_builder.source_to_idx:
        model_configs['vocab_size'] = len(data_builder.source_to_idx)

    if task_type == 'slu' and data_builder.class_to_idx:
        model_configs['class_size'] = len(data_builder.class_to_idx)

    if data_builder.target_to_idx:
        tag_to_idx = data_builder.target_to_idx

    if not test_only:
        model = create_model(task_type, tag_to_idx, model_configs)
        if 'load_model' in configs:
            logger.info('load model: {}'.format(configs['load_model']))
            if 'load_model_strict' in configs:
                strict = configs['load_model_strict']
            else:
                strict = False
            logger.info('set load model as strict method: {}'.format(strict))
            model = load_model(configs['load_model'], model, strict=strict)
        trainer = create_trainer(task_type, model, data_builder, train_configs,
                                 gpu_device=gpu_device, deploy_path=deploy_path / 'model')

        logger.info(model)
        trainer.train()

    if test_dataset_configs is not None:
        limit_len = test_dataset_configs['limit_len'] if 'limit_len' in test_dataset_configs else None

        best_model_path = deploy_path / 'model' / 'best_val.pkl'
        logger.info('load the best model')
        test_model = create_model(task_type, tag_to_idx, model_configs)
        test_model = load_model(best_model_path, test_model)
        evaluator = create_evaluator(task_type, test_model, data_builder, test_dataset_configs,
                                     limit_len)
        evaluator.eval()
        logger.info(evaluator.summary())


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    test_only = args.test_only
    main(configs, test_only)
