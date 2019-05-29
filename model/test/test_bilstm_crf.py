from model.bilstm_crf import BiLSTM_CRF
from model.utils import prepare_sequence
from configs.constants import *

EMBEDDING_DIM = 5
HIDDEN_DIM = 4


def test_run_model_predict():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)

    _, prediction = model(precheck_sent)

    assert len(prediction) == 11


def test_run_model_train():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tag = prepare_sequence(training_data[0][1], tag_to_ix)

    _, prediction = model.neg_log_likelihood(precheck_sent, precheck_tag)

    assert len(prediction) == 11
