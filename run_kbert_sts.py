# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from brain import KnowledgeGraph
from common.utils import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model

class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.criterion = nn.MSELoss()
        self.use_vm = False if args.no_vm else True
        logging.info(f"[BertClassifier] use visible_matrix: {self.use_vm}")

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(logits.view(-1), label.view(-1))
        return loss, logits



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--cache_path", default="./cache/stsb/cache.json", type=str,
                        help="Path of KG cache file.")
    parser.add_argument("--cache_embedding_path", default="./cache/stsb/cache_embeddings.json", type=str,
                        help="Path of embedding cache file.")
    parser.add_argument("--sentence_embedding_path", default="./cache/stsb/", type=str,
                        help="Path of dataset embedding file.")
    parser.add_argument("--logging_path", default="./outputs/logging/", type=str,
                        help="Path of logging file output.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=8,
                        help="Random seed.")

    # kg
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    # CPU switch
    parser.add_argument("--cpu", action="store_true", required=False, help="Strictly use CPU or not", default=False)

    # Connection URL for SQL DB
    parser.add_argument("--sqlconnectionurl", required=False, help="Connection URL for PostgreSQL database", default=None)

    # Get number of labels
    parser.add_argument("--labels_num", required=False, type=int, help="Number of classes/labels. If regression problem, set to 1", default=-1)
        
    parser.add_argument("--entity_recognition", choices=["spacy", "none"], default="spacy",
                        help="Specify entity extraction method" 
                             "Spacy NER"
                             "Do not inject knowledge"
                             )

    parser.add_argument("--threshold", required=False, type=float, help="Embedding similarity threshold for term insertion", default=0)

    parser.add_argument("--compute_embeddings", action="store_true", required=False, help="Compute new embeddings or use pre-computed", default=False)

    parser.add_argument("--dataset_cache_limit", type=int, default=6000, help="Split size of dataset cache embedding files")

    parser.add_argument("--sequence", type=int, help="Sequence to inject knowledge into", default=None)

    parser.add_argument("--max_seq_len", type=int, help="Maximum Sequence length where knowledge can be injected into", default=None)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    if args.sqlconnectionurl is not None:
        args.compute_embeddings = True

    if args.compute_embeddings and args.workers_num > 1:
        logging.info(f'Computation of embeddings must be done sequentially... setting workers_num to 1')
        args.workers_num = 1

    # run experiment
    start_time = time.perf_counter()
    run(args)
    end_time = time.perf_counter()
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

def run(args):

    set_seed(args.seed)
    set_tf_logging_level()

    logging_filename = f'{args.logging_path}kbert-sts-{time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())}.log'
    setup_logging(logging_filename)
    log_args(args)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                break
                # pass
    if args.labels_num == -1:
        args.labels_num = len(labels_set) 

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        logging.info(f"{torch.cuda.device_count()} GPUs are available. Let's use them.")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Build knowledge graph.
    kg = KnowledgeGraph(connurl=args.sqlconnectionurl, cache_path=args.cache_path,
     cache_embedding_path=args.cache_embedding_path, compute_embeddings=args.compute_embeddings)


    # Training phase.
    logging.info("Start training.")
    ss = time.perf_counter()
    trainset = read_dataset(args.train_path, columns, kg, vocab, args, workers_num=args.workers_num)
    ee = time.perf_counter()
    logging.info(f'Time taken to read training set: {ee-ss:.2f}s')
    logging.info("Shuffling dataset")

    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    # logging.info("Trans data to tensor.")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.FloatTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    vms = np.array([example[4] for example in trainset], dtype=np.uint8)

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"The number of training instances: {instances_num}")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    # fname = f"outputs/debug/token-output-{int(time.time())}.csv"
    logging.info('Begin training loop')
    for epoch in range(1, args.epochs_num + 1):
        train_loop_start = time.perf_counter()
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
            model.zero_grad()

            vms_batch = torch.LongTensor(np.array([vec_to_sym_matrix(vec, args.seq_length) for vec in vms_batch], dtype=np.uint8))

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                logging.info(f"Epoch id: {epoch}, Training steps: {i+1}, Avg loss: {total_loss / args.report_steps:.3f}")
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        train_loop_end = time.perf_counter()
        logging.info(f'Time taken for epoch {epoch} in training loop: {train_loop_end - train_loop_start:.2f}s')
        logging.info("Start evaluation on dev dataset.")
        eval_start = time.perf_counter()
        result = evaluate(model, device, args, False, columns, kg, vocab)
        if result > best_result:
            best_result = result
            save_model(model.cpu(), args.output_model_path)
            model = model.to(device)
        # else:
        #     continue
        eval_end = time.perf_counter()
        print(f'Evaluation on dev dataset time taken: {eval_end - eval_start:.2f}s')
        print("Start evaluation on test dataset.")
        test_start = time.perf_counter()
        evaluate(model, device, args, True, columns, kg, vocab)
        test_end = time.perf_counter()
        print(f'Evaluation on test dataset time taken: {test_end - test_start:.2f}s')

    if args.epochs_num == 1:
        return
    # Evaluation phase.
    logging.info("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(model, device, args, True, columns, kg, vocab)

if __name__ == "__main__":
    main()
