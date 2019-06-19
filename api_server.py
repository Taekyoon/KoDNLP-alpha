from fire import Fire

from flask import jsonify
from flask import Flask
from flask_cors import CORS

import numpy as np

from utils import set_gpu_device

set_gpu_device('-1')
app = Flask(__name__)
CORS(app)

tokenizer = None
model = None
index2intent = None
ner_tokenizer = None
graph = None


def ner_postprocessing(ner_tags, text):
    tag_stack = []
    entity_stack = []

    entities = []
    entity_tags = []

    tokenized_text = nltk.tokenize.WordPunctTokenizer().tokenize(text)
    text_for_ner = [t for t in tokenized_text if not t == "'"]

    for n, t in zip(ner_tags, text_for_ner):
        if not (n == 'o' or n == 'p'):
            state, tag = n.split('-')
            if state == 'b':
                if len(entity_stack) > 0:
                    entities.append(' '.join(entity_stack))
                    entity_tags.append(tag_stack[0])
                    entity_stack = []
                    tag_stack = []

                entity_stack.append(t)
                tag_stack.append(tag)

            elif state == 'i':
                entity_stack.append(t)
                tag_stack.append(tag)

        else:
            if len(entity_stack) > 0:
                entities.append(' '.join(entity_stack))
                entity_tags.append(tag_stack[0])
                entity_stack = []
                tag_stack = []

    if len(entity_stack) > 0:
        entities.append(' '.join(entity_stack))
        entity_tags.append(tag_stack[0])

    return entity_tags, entities


def model_process(text, tokenizer, index2intent, ner_tokenizer, model):
    input_seqs = preprocess(text, tokenizer)
    output_seqs = model.predict(input_seqs)
    pred_seqs = np.argmax(output_seqs[1][:, :, :], axis=-1)
    intent_confidence = round(float(max(output_seqs[0][0])), 3)
    intent_name = index2intent[np.argmax(output_seqs[0], axis=-1)[0]]
    ner_tags = [ner_tokenizer.index_word[s] for s in pred_seqs[0]]

    entity_tags, entities = ner_postprocessing(ner_tags, text)

    intent = {'intent_name': intent_name, 'confidence': intent_confidence}
    slots = [{'entity': tag, 'value': entity} for tag, entity in zip(entity_tags, entities)]

    json_item = {'intent': intent, 'slots': slots}

    return json_item


@app.route('/text_request/<query>')
def text_request(query):
    text = query

    with graph.as_default():
        query_analyze_response = model_process(text, tokenizer, index2intent, ner_tokenizer, model)

    text_response = manager.run(query_analyze_response)

    response = {'query_text': text,
                'query_analysis': query_analyze_response,
                'text_response': text_response}

    return jsonify(response)


def main(configs_path: str = './scripts/version_0_0_2_1_configs.json'):
    _configs = load_configs(configs_path)

    _model_path = _configs['deployment']['dir_path'] + _configs['deployment']['model_path']
    _data_manager_path = _configs['deployment']['dir_path'] + _configs['deployment']['data_manager_path']

    _custom_objects = {'loss': loss,
                       'precision': precision,
                       'recall': recall,
                       'f1score': f1score}

    data_manager = load_data_manager(_data_manager_path)

    global tokenizer
    global model
    global index2intent
    global ner_tokenizer
    global graph

    graph = tf.get_default_graph()

    tokenizer = data_manager.get_tokenizer()
    ner_tokenizer = data_manager._ner_tokenizer
    ner_tokenizer.index_word[0] = 'p'

    index2intent = data_manager._index_intent

    with graph.as_default():
        model = load_keras_model(_model_path, _custom_objects)

    app.run(host='0.0.0.0', port=10010, debug=True)


if __name__ == '__main__':
    Fire(main)
