from pathlib import Path

import torch

from flask import jsonify
from flask import Flask
from flask_cors import CORS

from data_manager.bert_tokenization.utils import create_bert_tokenizer
from data_manager.utils import load_vocab_dir
from model.utils import create_model
from postpro.ner import process_by_ner
from utils import set_gpu_device, parse_args, load_json, load_model

set_gpu_device('-1')
app = Flask(__name__)
CORS(app)

tokenizer_type = None
task_type = None
model = None
vocabs = None
bert_tokenizer = None


@app.route('/text_request/<query>')
def text_request(query):
    input_text = query

    if task_type == 'slu':
        if tokenizer_type == 'syllable_tokenizer':
            tokens = [ch for ch in input_text.replace(' ', '')]
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
        intent_prob = class_prob.tolist()[0][torch.argmax(class_prob, dim=-1).tolist()[0]]

        slot_tags, slots = process_by_ner(tokens, labeled_tag_seq)

        intent = {'intent': intent, 'prob': intent_prob}
        entities = [{'entity': tag, 'value': entity.replace(' ', '').replace('_', ' ').strip()} for tag, entity in
                    zip(slot_tags, slots)]

        raws = {'slot_tags': labeled_tag_seq, 'tokenized_text': ' '.join(tokens)}

        labels = {'intent': vocabs['class_vocab'].idx_to_word,
                  'slots': list(set([v.split('-')[-1] for v in vocabs['label_vocab'].idx_to_word[4:]]))}
        json_item = {'intent': intent, 'slots': entities, 'raws': raws, 'labels': labels}
    else:
        raise NotImplementedError()

    return jsonify(json_item)


def main(configs):
    global tokenizer_type
    global task_type
    global model
    global vocabs
    global bert_tokenizer

    task_type = configs['type']
    tokenizer_type = configs['tokenizer'] if 'tokenizer' in configs else ''
    deploy_path = Path(configs['deploy']['path'])
    model_configs = configs['model']
    best_model_path = deploy_path / 'model' / 'best_val.pkl'

    vocabs = load_vocab_dir(task_type, deploy_path)

    if 'input_vocab' in vocabs:
        model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

    if 'class_vocab' in vocabs:
        model_configs['class_size'] = len(vocabs['class_vocab'].word_to_idx)

    if 'label_vocab' in vocabs:
        tag_vocab = vocabs['label_vocab']

    model = create_model(task_type, tag_vocab, model_configs)
    model = load_model(best_model_path, model)
    model.eval()

    if tokenizer_type == 'bert_tokenizer':
        bert_tokenizer = create_bert_tokenizer()

    app.run(host='0.0.0.0', port=10010, debug=True)


if __name__ == '__main__':
    args = parse_args()
    configs = load_json(args.configs_path)
    main(configs)
