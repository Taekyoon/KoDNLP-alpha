from configs.constants import PAD, START_TAG, STOP_TAG

from trainer.seq_tag_trainer import SequenceTaggingModelTrainer
from trainer.seq_tag_cls_trainer import JointSequenceTagAndClassModelTrainer
from trainer.seq_pair_trainer import SequencePairModelTrainer


def create_trainer(type, model, data_builder, train_configs, gpu_device=-1, deploy_path='./tmp'):
    learning_rate = train_configs['learning_rate'] if 'learning_rate' in train_configs else 3e-4
    eval_batch_size = train_configs['eval_batch_size'] if 'eval_batch_size' in train_configs else 1

    if type == 'ner' or type == 'word_segment' or type == 'slu':
        train_data_loader, valid_data_loader = data_builder.build_data_loader(train_configs['batch_size'],
                                                                              train_configs['sequence_length'],
                                                                              valid_batch_size=eval_batch_size,
                                                                              enable_length=True)
    elif type == 'translate':
        train_data_loader, valid_data_loader = data_builder.build_data_loader(train_configs['batch_size'],
                                                                              train_configs['sequence_length'],
                                                                              train_configs['sequence_length'],
                                                                              valid_batch_size=eval_batch_size,
                                                                              enable_length=True)

    if type == 'ner' or type == 'word_segment':
        tag_vocabs = data_builder.target_vocab.idx_to_word
        tag_to_idx = data_builder.target_to_idx

        del tag_vocabs[tag_vocabs.index(PAD)]
        del tag_vocabs[tag_vocabs.index(START_TAG)]
        del tag_vocabs[tag_vocabs.index(STOP_TAG)]

        tag_inidices = [tag_to_idx[t] for t in tag_vocabs]

        trainer = SequenceTaggingModelTrainer(train_data_loader,
                                              valid_data_loader,
                                              model,
                                              train_configs['epochs'],
                                              train_configs['eval_steps'],
                                              learning_rate=learning_rate,
                                              deploy_path=deploy_path,
                                              gpu_device=gpu_device,
                                              eval_labels=tag_inidices)
    elif type == 'slu':
        tag_vocabs = data_builder.target_vocab.idx_to_word
        tag_to_idx = data_builder.target_to_idx

        del tag_vocabs[tag_vocabs.index(PAD)]
        del tag_vocabs[tag_vocabs.index(START_TAG)]
        del tag_vocabs[tag_vocabs.index(STOP_TAG)]

        tag_inidices = [tag_to_idx[t] for t in tag_vocabs]

        trainer = JointSequenceTagAndClassModelTrainer(train_data_loader,
                                                       valid_data_loader,
                                                       model,
                                                       train_configs['epochs'],
                                                       train_configs['eval_steps'],
                                                       deploy_path=deploy_path,
                                                       gpu_device=gpu_device,
                                                       eval_labels=tag_inidices)
    elif type == 'translate':
        trainer = SequencePairModelTrainer(train_data_loader,
                                           valid_data_loader,
                                           model,
                                           train_configs['epochs'],
                                           train_configs['eval_steps'],
                                           deploy_path=deploy_path,
                                           gpu_device=gpu_device)

    else:
        raise ValueError()

    return trainer
