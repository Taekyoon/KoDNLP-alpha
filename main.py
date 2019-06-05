from pathlib import Path

from utils import create_builder, create_model, create_trainer, create_evaluator, parse_args, load_json


def main(configs):
    task_type = configs['type']
    train_dataset_configs = configs['dataset']['train']
    test_dataset_configs = configs['dataset']['test']
    model_configs = configs['model']
    train_configs = configs['train']
    deploy_path = Path(configs['deploy']['path'])
    gpu_device = configs['gpu_device']

    data_builder = create_builder(task_type, train_dataset_configs, deploy_path=deploy_path)
    model = create_model(task_type, data_builder, model_configs)
    trainer = create_trainer(task_type, model, data_builder, train_configs,
                             gpu_device=gpu_device, deploy_path=deploy_path)
    evaluator = create_evaluator(task_type, model, data_builder, test_dataset_configs)

    print(model)
    trainer.train()
    evaluator.eval()
    print(evaluator.summary())


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    main(configs)
