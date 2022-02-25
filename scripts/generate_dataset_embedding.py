# # Building cache for dataset embeddings:
# ## - Embedding will be generated using RoBERTa for each sentence in dataset
# ## - Embeddings will be stored in json cache file
# ## - For datasets > LIMIT samples
# ##      - Split dataset into separate cache files of size LIMIT

import json
from tqdm import tqdm
from transformer import RoBERTa
import os
import argparse
import glob

LIMIT = 1000

embedder = RoBERTa()

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as dd:
        lines = [list(map(lambda x: x.replace('\n', ''), row.split('\t')[1:])) for row in dd]
    return lines[1:]

def fnv1a_64(string, seed=0):
    """
    Returns: The FNV-1a (alternate) hash of a given string
    """
    #Constants
    FNV_prime = 1099511628211
    offset_basis = 14695981039346656037

    #FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash ^ ord(char)
        hash = hash * FNV_prime
    return hash

def compute_hash(x, nbits=16):
    hash = fnv1a_64(x)
    hash_str = str(hash)
    hash_short = int(hash_str[:nbits] + hash_str[-nbits:])
    return hash_short

def write_to_file(path, ob):
    with open(path, 'w', encoding='utf-8') as ff:
        ff.write(json.dumps(ob))

def create_folder(base, x):
    folder_name = os.path.join(base, hex(compute_hash(x)))
    if not os.path.exists(base):
        os.makedirs(base)
    return folder_name

def create_embedding_cache(dataset, base, extension='json', limit=LIMIT):
    cache = {}
    output_folder = create_folder(base, " ".join(dataset[0]))
    for i, sent_pair in tqdm(enumerate(dataset)):
        if i % limit == 0:
            if len(cache.keys()) > 0:
                write_to_file(f'{output_folder}.{extension}', cache)

            output_folder = create_folder(base, " ".join(sent_pair))
            cache.clear()
        for sent in sent_pair:
            if sent not in cache:
                sent_embedding = embedder.get_sentence_embedding(sent).squeeze()
                cache[sent.strip()] = sent_embedding.tolist()
    if len(cache.keys()) > 0:
        write_to_file(f'{output_folder}.{extension}', cache)


def create_dataset_embedding_cache(dataset_paths, output_path):
    for path in dataset_paths:
        dataset = load_dataset(path)
        print(f'Generating cache for {path}')
        create_embedding_cache(dataset, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help=['base path for dataset'])
    parser.add_argument('--extension', help=['extension for dataset files'], default='tsv')
    parser.add_argument('--output_path', help=['base output path'], default='./')
    args = parser.parse_args()
    files = os.path.join(args.dataset_path, f'*.{args.extension}')
    file_paths = glob.glob(files)
    create_dataset_embedding_cache(file_paths, args.output_path)
    print(f'Cache successfully created in {args.output_path}')
if __name__ == "__main__":
    main()