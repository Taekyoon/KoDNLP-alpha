from model.sequence_tagger.bilstm_crf import BilstmCRF
from model.joint_classifier_and_sequence_tagger.bilstm_crf import BilstmCRF as BilstmCRF_SLU


def create_crf_model(type, tag_to_idx, model_configs):
    model_type = model_configs['type']
    model_params = model_configs['parameters']
    if type == 'ner':
        vocab_size = model_configs['vocab_size']
        if model_type == 'bilstm_crf':
            model = BilstmCRF(vocab_size, tag_to_idx, model_params['word_embedding_dims'],
                               model_params['hidden_dims'])
        else:
            raise ValueError()
    elif type == 'slu':
        vocab_size, class_size = model_configs['vocab_size'], model_configs['class_size']

        # num_layers = model_params['lstm_num_layers'] if 'lstm_num_layers' in model_params else 1
        # dropout = model_params['lstm_dropout'] if 'lstm_dropout' in model_params else 0.5

        if model_type == 'bilstm_crf':
            model = BilstmCRF_SLU(vocab_size, class_size, tag_to_idx, model_params['word_embedding_dims'],
                              model_params['hidden_dims'])
        else:
            raise ValueError()
    else:
        raise ValueError()

    return model
