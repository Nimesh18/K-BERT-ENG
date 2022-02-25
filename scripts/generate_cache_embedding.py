import json
from transformer import RoBERTa
import argparse

embedder = RoBERTa()

def load_cache(path):
    with open(path, encoding='utf-8') as fd:
        return json.loads(fd.readline())

def create_embeddings_for_cache(cache_path, output):
    cache = load_cache(cache_path)
    embeddings = {}
    for key, value in cache.items():
        if len(value) > 0:
            instance_embeddings = embedder.get_sentence_embedding(list(map(lambda x: " ".join(x), value)))
            embeddings[key] = instance_embeddings.tolist()

    with open(output, 'w', encoding='utf-8') as kk:
        kk.write(json.dumps(embeddings))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_file', help=['path of cache file'])
    parser.add_argument('--output_file', help=['output file'], default='./')
    args = parser.parse_args()

    create_embeddings_for_cache(args.cache_file, args.output_file)
    print(f'Cache successfully created in {args.output_file}')

if __name__ == "__main__":
    main()