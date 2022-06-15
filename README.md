# K-BERT-ENG
![](https://img.shields.io/badge/license-MIT-000000.svg)

Sorce code and datasets for model based on ["K-BERT: Enabling Language Representation with Knowledge Graph"](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuW.5594.pdf), which is implemented based on the [UER](https://github.com/dbiir/UER-py) framework.


## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
transformers
sentence_transformers
spacy==3.0.0
```


## Prepare
All required data can be found [here]()
* Download the ``bert-base-uncased.bin`` and save it to the ``models/english`` directory.
* Download the ``cache`` folder and save it in root ``/`` directory.
* Download the ``datasets`` folder and save it in root ``/`` directory.
* Optional - Download  ``Wikidata.pgsql`` postgreSQL dump and restore in postgreSQL DB.


## K-BERT for STS-B

### STS example

```sh
python3 K-BERT-master/run_kbert_sts.py \
    --pretrained_model_path ./K-BERT-master/models/english/bert-base-uncased.bin \
    --output_model_path ./K-BERT-master/outputs/kbert_Wikidata_stsb.bin \
    --config_path ./K-BERT-master/models/google_config.json \
    --vocab_path ./K-BERT-master/models/english/vocab-uncased.txt \
    --train_path ./K-BERT-master/datasets/stsb/original/train.tsv \
    --dev_path ./K-BERT-master/datasets/stsb/original/dev.tsv \
    --test_path ./K-BERT-master/datasets/stsb/original/test.tsv \
    --cache_path ./K-BERT-master/cache/stsb/cache.json \
    --cache_embedding_path ./K-BERT-master/cache/stsb/cache_embeddings.json \
    --sentence_embedding_path ./K-BERT-master/cache/stsb/ \
    --logging_path ./K-BERT-master/outputs/logging/ \
    --epochs_num 10 --batch_size 16 --labels_num 1 --learning_rate 4e-05 \
    --workers_num 1 --entity_recognition "spacy" --threshold 0.5 --seed 8
```

Options of ``run_kbert_sts.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--output_model_path] - Path to the output model.
        [--cache_path] - Path of the KG cache file.
        [--cache_embedding_path] - Path of the cache embedding file.
        [--sentence_embedding_path] - Path of the dataset embedding file.
        [--logging_path] - Path to output logs.
        [--correct_ents_path] - Path of the manually injected knowledge.
        [--manual] - Perform Manual Knowledge Injection.
        [--entity_recognition] - Entity extraction method.
        [--threshold] - Similarity threshold between embeddings.
        [--max_seq_len] - Maximum sequence length permittable for knowledge injection.
        [--dataset_cache_limit] - Number of items per dataset cache file.
```

## K-BERT for text classification

### AG NEWS SUBSET example

```
python3 K-BERT-master/run_kbert_cls.py \
    --pretrained_model_path ./K-BERT-master/models/english/bert-base-uncased.bin \
    --output_model_path ./K-BERT-master/outputs/kbert_Wikidata_cls.bin \
    --config_path ./K-BERT-master/models/google_config.json \
    --vocab_path ./K-BERT-master/models/english/vocab-uncased.txt \
    --train_path ./K-BERT-master/datasets/ag_news/original/train.tsv \
    --dev_path ./K-BERT-master/datasets/ag_news/original/dev.tsv \
    --test_path ./K-BERT-master/datasets/ag_news/original/test.tsv \
    --cache_path ./K-BERT-master/cache/ag_news/cache.json \
    --cache_embedding_path ./K-BERT-master/cache/ag_news/cache_embeddings.json \
    --sentence_embedding_path ./K-BERT-master/cache/ag_news/ \
    --logging_path ./K-BERT-master/outputs/logging/ \
    --epochs_num 10 --batch_size 32 --seq_length 128 --learning_rate 5e-05 \
    --workers_num 1 --entity_recognition "spacy" --threshold 0.6 --dataset_cache_limit 6000 --seed 10 
```

Options of ``run_kbert_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--output_model_path] - Path to the output model.
        [--cache_path] - Path of the KG cache file.
        [--cache_embedding_path] - Path of the cache embedding file.
        [--sentence_embedding_path] - Path of the dataset embedding file.
        [--logging_path] - Path to output logs.
        [--correct_ents_path] - Path of the manually injected knowledge.
        [--manual] - Perform Manual Knowledge Injection.
        [--entity_recognition] - Entity extraction method.
        [--threshold] - Similarity threshold between embeddings.
        [--max_seq_len] - Maximum sequence length permittable for knowledge injection.
        [--dataset_cache_limit] - Number of items per dataset cache file.
```
