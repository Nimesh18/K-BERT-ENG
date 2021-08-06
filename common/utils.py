import sys
import torch
import torch.nn as nn
from uer.utils.constants import *
from multiprocessing import Process, Pool, cpu_count
import spacy


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

    p_id, sentences, named_entities, columns, kg, vocab, args = params
    # print(columns)
    sentences_num = len(sentences)
    dataset = []
    for line_id, (line, entities) in enumerate(zip(sentences, named_entities)):
        if line_id % 1000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]
   
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], entities[1:], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0]

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))
            
            elif len(line) == 3:
                # for sts normalize to between 0 and 1 by dividing by 5
                label = float(line[columns["label"]])/5.0 if args.labels_num == 1 else int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], entities[1:], add_pad=True, max_length=args.seq_length)
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
            
            elif len(line) == 4:  # for dbqa
                qid=int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], entities[1:], add_pad=True, max_length=args.seq_length)
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
                
                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass
            
        except Exception as e:
            print("Error line: ", line, e)
            raise ValueError
    return dataset


def read_dataset(path, columns, kg, vocab, args, workers_num=1):

        # Spacy setup
        nlp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        # processed = nlp.pipe(sentences, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        processed = nlp.pipe(sentences)

        named_entities = list(map(lambda x: [a.text for a in x.ents], list(processed)))

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], 
                named_entities[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, named_entities, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset


# Evaluation function.
def evaluate(model, device, args, is_test, columns, kg, vocab, metrics='Acc'):
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
    vms = [example[4] for example in dataset]

    batch_size = args.batch_size
    instances_num = input_ids.size()[0]
    if is_test:
        print("The number of evaluation instances: ", instances_num)

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    model.eval()
    
    if args.labels_num == 1:
        total_mse_loss= 0
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                except:
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            total_mse_loss += loss.item()
            # print(f'Loss for batch {i} is {loss.item()}')
        print(f"Total MSE Loss = {total_mse_loss:.3f}, batches = {i + 1}, Avg loss = {total_mse_loss/(i+1):.3f}")
        return (i + 1) / total_mse_loss if total_mse_loss > 0 else float('inf')

    elif args.mean_reciprocal_rank:
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
            logits = nn.Softmax(dim=1)(logits)
            if i == 0:
                logits_all=logits
            if i >= 1:
                logits_all=torch.cat((logits_all,logits),0)
    
        order = -1
        gold = []
        for i in range(len(dataset)):
            qid = dataset[i][-1]
            label = dataset[i][1]
            if qid == order:
                j += 1
                if label == 1:
                    gold.append((qid,j))
            else:
                order = qid
                j = 0
                if label == 1:
                    gold.append((qid,j))

        label_order = []
        order = -1
        for i in range(len(gold)):
            if gold[i][0] == order:
                templist.append(gold[i][1])
            elif gold[i][0] != order:
                order=gold[i][0]
                if i > 0:
                    label_order.append(templist)
                templist = []
                templist.append(gold[i][1])
        label_order.append(templist)

        order = -1
        score_list = []
        for i in range(len(logits_all)):
            score = float(logits_all[i][1])
            qid=int(dataset[i][-1])
            if qid == order:
                templist.append(score)
            else:
                order = qid
                if i > 0:
                    score_list.append(templist)
                templist = []
                templist.append(score)
        score_list.append(templist)

        rank = []
        pred = []
        print(len(score_list))
        print(len(label_order))
        for i in range(len(score_list)):
            if len(label_order[i])==1:
                if label_order[i][0] < len(score_list[i]):
                    true_score = score_list[i][label_order[i][0]]
                    score_list[i].sort(reverse=True)
                    for j in range(len(score_list[i])):
                        if score_list[i][j] == true_score:
                            rank.append(1 / (j + 1))
                else:
                    rank.append(0)

            else:
                true_rank = len(score_list[i])
                for k in range(len(label_order[i])):
                    if label_order[i][k] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][k]]
                        temp = sorted(score_list[i],reverse=True)
                        for j in range(len(temp)):
                            if temp[j] == true_score:
                                if j < true_rank:
                                    true_rank = j
                if true_rank < len(score_list[i]):
                    rank.append(1 / (true_rank + 1))
                else:
                    rank.append(0)
        MRR = sum(rank) / len(rank)
        print("MRR", MRR)
        return MRR

    else:
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                except:
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
    
        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")
        
        for i in range(confusion.size()[0]):
            p = confusion[i,i].item()/confusion[i,:].sum().item()
            r = confusion[i,i].item()/confusion[:,i].sum().item()
            f1 = 2*p*r / (p+r)
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
        if metrics == 'f1':
            return label_1_f1
        else:
            return correct/len(dataset)


# import csv
def convert_ids_to_string(ids, tokenizer, filename="input_ids_string.csv"):
    with open(filename, mode='a', encoding='utf-8') as fd:
        # for id in ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        token_strings = tokenizer.convert_tokens_to_string(tokens)
        fd.write(token_strings + "\n")