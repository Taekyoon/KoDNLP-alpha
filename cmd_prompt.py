import torch

from pathlib import Path
import logging
from prettytable import PrettyTable

from data_manager.utils import load_vocab
from data_manager.bert_tokenization.utils import create_bert_tokenizer
from model.utils import create_crf_model
from postpro.ner import process_by_ner

from utils import parse_args, load_json, load_model

logger = logging.getLogger(__name__)


def main(configs):
    task_type = configs['type']
    tokenizer_type = configs['tokenizer'] if 'tokenizer' in configs else ''
    deploy_path = Path(configs['deploy']['path'])
    model_configs = configs['model']
    best_model_path = deploy_path / 'model' / 'best_val.pkl'

    vocabs = load_vocab(task_type, deploy_path)

    if 'input_vocab' in vocabs:
        model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

    if 'class_vocab' in vocabs:
        model_configs['class_size'] = len(vocabs['class_vocab'].word_to_idx)

    if 'label_vocab' in vocabs:
        tag_vocab = vocabs['label_vocab']

    model = create_crf_model(task_type, tag_vocab, model_configs)
    model = load_model(best_model_path, model)
    model.eval()

    while True:
        input_text = input('input text: ')

        if input_text == 'quit':
            break

        if task_type == 'slu':
            if tokenizer_type == 'syllable_tokenizer':
                tokens = [ch for ch in input_text.replace(' ', '$')]
            elif tokenizer_type == 'bert_tokenizer':
                tokenizer = create_bert_tokenizer()
                tokens = tokenizer.tokenize(input_text)
            elif tokenizer_type == 'space_tokenizer':
                tokens = input_text.split()

            model_inputs = vocabs['input_vocab'].to_indices(tokens)

            model_inputs = torch.Tensor([model_inputs]).long()
            pred_score, tag_seq, class_prob = model(model_inputs)

            labeled_tag_seq = vocabs['label_vocab'].to_tokens(tag_seq[0].tolist())

            intent = vocabs['class_vocab'].to_tokens(torch.argmax(class_prob, dim=-1).tolist())[0]

            slot_tags, slots = process_by_ner(tokens, labeled_tag_seq)

            slot_table = PrettyTable(['Slot', 'Type'])

            for s, t in zip(slots, slot_tags):
                entity = s
                if tokenizer_type == 'syllable_tokenizer':
                    entity = s.replace(' ', '')
                elif tokenizer_type == 'bert_tokenizer':
                    entity = s.replace(' ', '').replace('_', ' ').strip()
                    # if entity[-1] == '_':
                    #     entity = entity[:-1]

                slot_table.add_row([entity, t])

            print('*' * 5, 'Intents', '*' * 5)
            print(intent, class_prob.tolist()[0][torch.argmax(class_prob, dim=-1).tolist()[0]])
            print('*' * 5, 'Slots', '*' * 5)
            print(pred_score.item() / len(model_inputs[0]))
            print(slot_table)
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    main(configs)
