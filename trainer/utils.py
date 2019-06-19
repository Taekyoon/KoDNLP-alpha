from trainer.ner_trainer import NERModelTrainer
from trainer.slu_trainer import SLUModelTrainer


def create_trainer(type, model, data_builder, train_configs, gpu_device=-1, deploy_path='./tmp'):
    train_data_loader, valid_data_loader = data_builder.build_data_loader(train_configs['batch_size'],
                                                                          train_configs['sequence_length'],
                                                                          enable_length=True)

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
