import os
import csv
import json
import spacy
import time
import logging
import torch
from torch.nn import Softmax
from uer.utils.constants import *
from multiprocessing import Pool
import numpy as np
from brain.config import NEVER_SPLIT_TAG, DATASET_CACHE_LIMIT
from scipy.stats import spearmanr, pearsonr

def allowed_ner_labels(nlp):
    all_labels = nlp.get_pipe('ner').labels
    to_remove = ['CARDINAL', 'DATE', 'PERCENT', 'TIME', 'MONEY', 'LAW', 'QUANTITY', 'ORDINAL']
    return set(all_labels) - set(to_remove)

def distribute_sentences(sentences, named_entities, num_workers, cap=DATASET_CACHE_LIMIT):
    grouped_sents = []
    grouped_ents = []
    for offset in range(num_workers):
        indicies = [i + offset for i in range(0, len(sentences) - offset, num_workers)]
        max_idx = indicies[-1]
        grouped_indicies = []
        curr_idxs = []
        if max_idx >= cap:
            mul = 1
            for idx in indicies:
                if idx >= mul * cap:
                    mul += 1
                    grouped_indicies.append(curr_idxs)
                    curr_idxs = []
                curr_idxs.append(idx)
            if len(curr_idxs) > 0:
                grouped_indicies.append(curr_idxs)
        else:
            grouped_indicies = [indicies]
        
        grouped_sents.append([[sentences[idx] for idx in idxs] for idxs in grouped_indicies])
        grouped_ents.append([[named_entities[idx] for idx in idxs] for idxs in grouped_indicies])
    return grouped_sents, grouped_ents, len(grouped_indicies)

def get_sentence_embedding_files(sentences, embedding_path, cap=DATASET_CACHE_LIMIT, extension = 'json'):
    folder_names = []
    for i in range(0, len(sentences), cap):
        line = " ".join(list(map(lambda x: x.replace('\n',''), sentences[i].split('\t')))[1:])
        folder_name = hex(compute_hash(line))
        fullpath = os.path.join(embedding_path, f'{folder_name}.{extension}')
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f'sentence embedding path not found for line: {line}')
        folder_names.append(fullpath)
    return folder_names

# Datset loader.
def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
    instances_num = input_ids.size()[0]
    for i in range(instances_num // batch_size):
        input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
        label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
        mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
        pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
        vms_batch = vms[i*batch_size: (i+1)*batch_size]
        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
    if instances_num > instances_num // batch_size * batch_size:
        input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
        label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
        mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
        pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
        vms_batch = vms[instances_num//batch_size*batch_size:]

        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch


def add_knowledge_worker(params):

    p_id, sentences, named_entities, columns, kg, vocab, args, e = params
    
    sentences_num = len(sentences)
    progress_step = int(sentences_num * 0.25)
    dataset = []
    for line_id, (line, entities) in enumerate(zip(sentences, named_entities)):
        # if line_id % progress_step == 0:
        #     logging.info(f"Progress of process {p_id}: {line_id}/{sentences_num}")
        #     sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]
   
                tokens, pos, vm = kg.add_knowledge_with_vm([text], entities, seq=args.sequence, max_seq_len=args.max_seq_len, max_length=args.seq_length, threshold=args.threshold, e=e)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0]

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))
            
            elif len(line) == 3:
                # for sts normalize to between 0 and 1 by dividing by 5
                label = float(line[columns["label"]])/5.0 if args.labels_num == 1 else int(line[columns["label"]])
                text = [line[columns["text_a"]], line[columns["text_b"]]]

                tokens, pos, vm = kg.add_knowledge_with_vm([text], entities, seq=args.sequence, max_seq_len=args.max_seq_len, max_length=args.seq_length, threshold=args.threshold, e=e)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0]

                token_ids = [vocab.get(t) for t in tokens]

                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))
            else:
                pass
            
        except Exception as e:
            logging.error(f"Error line: {line} {e}")
            raise ValueError
    return dataset

def read_dataset(path, columns, kg, vocab, args, workers_num=1):

        logging.info(f"Loading sentences from {path}")
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)


        start_time = time.perf_counter()
        named_entities = get_entities(sentences, args)
        end_time = time.perf_counter()
        logging.info(f'Time taken for processing entities: {end_time - start_time:.2f}s')

        use_cache = args.entity_recognition != "none" and not args.compute_embeddings
        if use_cache:
            embedding_paths = get_sentence_embedding_files(sentences, args.sentence_embedding_path, args.dataset_cache_limit)
        sents, ents, no_pools = distribute_sentences(sentences, named_entities, workers_num, args.dataset_cache_limit)

        logging.info(f"There are {sentence_num} sentence in total. We use {workers_num} processes to inject knowledge into sentences.")
        dataset = []
        if workers_num > 1:
            for p in range(no_pools):
                e = None if not use_cache else load_cache(embedding_paths[p])
                params = []
                for i in range(workers_num):
                    params.append((i, sents[i][p], ents[i][p], columns, kg, vocab, args, e))
                pool = Pool(workers_num)
                res = pool.map(add_knowledge_worker, params)
                pool.close()
                pool.join()
                dataset.extend([sample for block in res for sample in block])
        else:
            for p in range(no_pools):
                e = None if not use_cache else load_cache(embedding_paths[p])
                params = (0, sents[0][p], ents[0][p], columns, kg, vocab, args, e)
                dataset.extend(add_knowledge_worker(params))

        return dataset

def get_entities(sentences, args):

    if args.entity_recognition == "spacy":
        # Spacy setup
        nlp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        
        sentences = list(map(lambda x: x[x.index("\t"):], sentences))
        processed = nlp.pipe(sentences)

        named_entities = list(map(lambda x: [a.text for a in x.ents if a.label_ in allowed_ner_labels(nlp)], list(processed)))

    else:
        named_entities = list(map(lambda x: [], sentences))


    named_entities = [list(filter(lambda x: not x.isdigit(), group)) for group in named_entities]

    return named_entities

def print_named_entity_stats(named_entities):
    lens = list(map(lambda x: len(x), named_entities))
    logging.info('entity stats:')
    logging.info(f'max len: {max(lens)}')        
    logging.info(f'min len: {min(lens)}')        
    logging.info(f'average len: {sum(lens)/len(lens)}')        

# Evaluation function.
def evaluate(model, device, args, is_test, columns, kg, vocab):
    if is_test:
        dataset = read_dataset(args.test_path, columns, kg, vocab, args, workers_num=args.workers_num)
    else:
        dataset = read_dataset(args.dev_path, columns, kg, vocab, args, workers_num=args.workers_num)
    

    input_ids = torch.LongTensor([sample[0] for sample in dataset])
    if args.labels_num == 1:
        label_ids = torch.FloatTensor([sample[1] for sample in dataset]) 
    else:
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
    mask_ids = torch.LongTensor([sample[2] for sample in dataset])
    pos_ids = torch.LongTensor([example[3] for example in dataset])
    vms = np.array([example[4] for example in dataset], dtype=np.uint8)

    batch_size = args.batch_size
    instances_num = input_ids.size()[0]
    if is_test:
        logging.info(f"The number of evaluation instances: {instances_num}")

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    model.eval()
    
    if args.labels_num == 1:
        total_loss= 0
        all_labels = []
        all_logits = []
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            vms_batch = torch.LongTensor(np.array([vec_to_sym_matrix(vec, args.seq_length) for vec in vms_batch], dtype=np.uint8))

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            all_labels.extend(label_ids_batch.flatten().tolist())
            all_logits.extend(logits.flatten().tolist())

        spearman = spearmanr(all_labels, all_logits)
        pearsonc, pearsonp = pearsonr(all_labels, all_logits)
        logging.info(f"Total Loss = {total_loss:.3f}, batches = {i + 1}, Avg loss = {total_loss/(i+1):.3f}")
        logging.info(f"Spearman correlation = ({spearman.correlation:.5f}, {spearman.pvalue:.2f}), Pearson correlation = ({pearsonc:.5f}, {pearsonp:.2f})")

        return (i+1) / total_loss if total_loss > 0 else float('inf')


    else:
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            vms_batch = torch.LongTensor(np.array([vec_to_sym_matrix(vec, args.seq_length) for vec in vms_batch], dtype=np.uint8))

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)

            logits = Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

        if is_test:
            logging.info("Confusion matrix:")
            logging.info(confusion)
            logging.info("Report precision, recall, and f1:")
        
        for i in range(confusion.size()[0]):
            p = confusion[i,i].item()/confusion[i,:].sum().item()
            r = confusion[i,i].item()/confusion[:,i].sum().item()
            f1 = 2*p*r / (p+r)
            logging.info(f"Label {i}: {p:.3f}, {r:.3f}, {f1:.3f}")
        logging.info(f"Acc. (Correct/Total): {correct/instances_num:.4f} ({correct}/{instances_num})")
        return correct/instances_num


# import csv
def convert_ids_to_string(ids, tokenizer, filename="input_ids_string.csv"):
    with open(filename, mode='a', encoding='utf-8') as fd:
        # for id in ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        token_strings = tokenizer.convert_tokens_to_string(tokens)
        fd.write(token_strings + "\n")


def convert_ids_to_text(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    token_strings = tokenizer.convert_tokens_to_string(tokens)
    return token_strings

def write_input_batch_to_file(input_batch, tokenizer, filename):
    with open(filename, 'a', encoding='UTF-8') as fd:
        for ids in input_batch:
            str_out = convert_ids_to_text(ids, tokenizer)
            fd.write(str_out + "\n")

def write_loss_to_file(loss, filename):
    with open(filename, 'a') as ff:
        ff.write("=============\n" + str(loss) + "\n")

def write_output_to_csv(input_batch, tokenizer, logits, label_ids, filename):
    """
    write output to csv for manual inspection of predicted vs actual per sent
    """

    # batched = torch.dstack((input_batch, label_ids, logits))
    with open(filename, 'w', newline='', encoding='UTF-8') as wr:
        writer = csv.DictWriter(wr, fieldnames=['sent', 'label', 'predicted'], quoting=csv.QUOTE_NONE, dialect=csv.excel_tab, escapechar='/')
        writer.writeheader()
        for ids, label, pred in zip(input_batch, label_ids, logits):
            sentence = convert_ids_to_text(ids, tokenizer)
            writer.writerow({'sent': sentence, 'label': round(label.item(),3), 'predicted':round(pred.item(),3)})

def write_cls_outcome(filename, tokenizer, input_ids_batch, idxs, predictions, true_labels):
    with open(filename, 'a', encoding='UTF-8') as fd:
        for ids, idx, pred, gold in zip(input_ids_batch, idxs, predictions, true_labels):
            str_out = convert_ids_to_text(ids, tokenizer).replace('[PAD]', '').strip()
            fd.write(f"{idx}\t{gold}\t{pred}\t{str_out}\n")

def sym_matrix_to_vec(mat):
    dim = mat.shape[0]
    iu = np.mask_indices(dim, np.tril)
    vec = mat[iu]
    return vec

def vec_to_sym_matrix(vec, dim):
    indicies = np.tril_indices(dim)
    mat = np.zeros((dim, dim), dtype=vec.dtype)
    mat[indicies] = vec
    mat = np.tril(mat) + np.tril(mat, -1).T
    return mat

def set_tf_logging_level():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def setup_logging(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for hlr in logger.handlers:
        logger.removeHandler(hlr)
        
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    file = logging.FileHandler(filename,mode='a', encoding='utf-8')
    format = logging.Formatter("%(message)s")
    file.setLevel(logging.INFO)
    file.setFormatter(format)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(format)

    logger.addHandler(file)
    logger.addHandler(stream)

def log_args(args):

    logging.info(f"pretrained_model_path: {args.pretrained_model_path}")
    logging.info(f"config_path: {args.config_path}")
    logging.info(f"vocab_path: {args.vocab_path}")
    logging.info(f"output_model_path: {args.output_model_path}")
    logging.info(f"train_path: {args.train_path}")
    logging.info(f"dev_path: {args.dev_path}")
    logging.info(f"test_path: {args.test_path}")
    logging.info(f"sentence_embedding_path: {args.sentence_embedding_path}")
    logging.info(f"cache_path: {args.cache_path}")
    logging.info(f"cache_embedding_path: {args.cache_embedding_path}")
    logging.info(f"sqlconnectionurl: {args.sqlconnectionurl}, sequence: {args.sequence}, max_seq_len: {args.max_seq_len}")
    logging.info(f"pooling: {args.pooling}, cpu: {args.cpu}, no_vm: {args.no_vm}, labels_num: {args.labels_num}, manual: {args.manual}")
    logging.info(f"dropout: {args.dropout}, entity_recognition: {args.entity_recognition}, threshold: {args.threshold}, learning rate: {args.learning_rate}")
    logging.info(f"batch_size: {args.batch_size}, seq_length: {args.seq_length}, epochs_num: {args.epochs_num}, seed: {args.seed}")

def sort_by_len(ls):
    return sorted(ls, key=len, reverse=True)

def group_related_tokens(sent, tokenized_sent, named_entities):
    """
    sent: ["abcd", "abcd_pair"] i.e. sentence and it's pair
    returns:
    list of related tokens grouped, 
    indices of which group is an entity match,
    named entity corresponding to said index
    """
    entities = sort_by_len(named_entities)
    groups = []
    idx = -1
    tokens = tokenized_sent.tokens()
    entity_match_index = []
    matches = []
    while idx < len(tokens) - 1:
        idx += 1
        token = tokens[idx]
        group = [token]
        if token in NEVER_SPLIT_TAG:
            groups.append(group)
            continue
        char_span = tokenized_sent.token_to_chars(idx)
        sequence = tokenized_sent.token_to_sequence(idx)
        for entity in entities:
            lower_ent = entity.strip().lower().replace("_", " ")
            end_pos = char_span.start + len(lower_ent)
            if len(sent[sequence]) < end_pos:
                continue

            substr = sent[sequence][char_span.start: end_pos]
            if substr.lower() == lower_ent:
                token_end = tokenized_sent.char_to_token(end_pos - 1, sequence_index=sequence)
                group = [t for t in tokenized_sent.tokens()[idx: token_end + 1]]
                idx = token_end
                entity_match_index.append(len(groups))
                matches.append(entity)
                break # we only care about the longest entity match
        groups.append(group)
    return groups, entity_match_index, matches


def load_cache(path):
    with open(path, encoding='utf-8') as fd:
        return json.loads(fd.readline())


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