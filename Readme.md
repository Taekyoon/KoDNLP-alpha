## Korean Deep NLP Framework (KoDNLP)
Placeholder

### Requirements

- Python 3.6
- Pytorch
- Numpy
- Pandas
- Scikit Learn

### How to Run

##### Train NER Model

```bash
python main.py -c scripts/ner_bilstm_configs.json
``` 

##### Train SLU Model
```bash
python main.py -c scripts/snips_slu_bilstm_configs.json
```

### Test
```bash
pip install pytest
pytest
```
