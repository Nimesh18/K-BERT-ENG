# coding: utf-8
"""
KnowledgeGraph
"""
import os
from re import search, escape

from numpy.lib.arraysetops import isin
import brain.config as config
import pkuseg
import numpy as np
from database import Database
from transformers import BertTokenizerFast
from common.embedding.transformer import RoBERTa
from torch.utils.data import DataLoader


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, connurl, predicate=False):
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.connurl = connurl
        self.predicate = predicate
        
        if all(e not in spo_files for e in config.ENGLISH_KGS):
            print('using pkuseg')
            self.lookup_table = self._create_lookup_table()
            self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
            self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        else:
            print('using BertTokenizerFast')
            self.lookup_table = Database(connurl, predicate, spo_files)
            self.segment_vocab = config.NEVER_SPLIT_TAG
            self.lookup_table = None
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', never_split=self.segment_vocab)
            self.embedder = RoBERTa()
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def join_relevant_tokens(self, ch, sent, prefix='##'):
        if sent[ch].startswith(prefix) or ch + 1 >= len(sent):
            return sent[ch]
        acc = [sent[ch]]
        while ch < len(sent) - 1 and sent[ch + 1].startswith(prefix):
            acc.append(sent[ch + 1].replace(prefix, ''))
            ch+=1
        return ''.join(acc)


    def add_knowledge_with_vm(self, sent_batch, named_entities, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., [["abcd", "abcd_pair"], ["efgh", "efgh_pair"]]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        self.lookup_table = Database(self.connurl, self.predicate, self.spo_file_paths)
        split_sent_batch = [self.tokenizer(sent, text_pair=sent_pair) for sent, sent_pair in sent_batch]
        # split_entity_batch = [self.tokenizer.tokenize(named_entity) for named_entity in named_entities]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for sent, tokenized_sent in zip(sent_batch, split_sent_batch):

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            charspans = RoBERTa.get_character_spans(tokenized_sent)
            token_idx = 0

            for idx, token in enumerate(tokenized_sent.tokens()):

                # if isinstance(self.lookup_table, Database):
                limit = 2 ** 11
                offset = 0
                entities = []
                maximum_values = []
                maximum_ents = []
                more_entities = True
                sequence = None
                word_embedding = None
                tokenized_sent['input_ids'][idx]
                if token not in self.special_tags:
                    entity = self.join_relevant_tokens(idx, tokenized_sent.tokens())
                    matches = []
                    for entity_idx, named_entity in enumerate(named_entities):
                        escaped = escape(entity)
                        mt = search(rf'\b{escaped}\b', named_entity)
                        if mt is not None:
                            matches.append(entity_idx)
                    # matches = [(search(rf'\b{entity}\b', es), group_idx, es_idx) for group_idx, group in enumerate(named_entities) for es_idx, es in enumerate(group)]
                    if len(entity) > 1 and not entity.isdigit() and len(matches) > 0:
                        # print('in here')
                        entity = "_".join(named_entities[matches[0]].split())
                        while more_entities:
                            entities = list(self.lookup_table.search(entity, limit, offset))
                            offset +=limit
                            num_entities=len(entities)
                            if num_entities > 0:
                                dataloader = DataLoader(entities, batch_size=16)
                                sentence_embeddings = []
                                for e_batch in dataloader:
                                    sentence_embeddings.extend(self.embedder.get_sentence_embedding(e_batch))
                                if sequence is None or word_embedding is None:
                                    sequence = tokenized_sent.token_to_sequence(idx)
                                    word_embedding = self.embedder.get_embedding(token_idx, charspans, sent[sequence])
                                max_idx, max_val = RoBERTa.get_most_similar(word_embedding, sentence_embeddings) # put max_entities in here
                                maximum_ents.append(entities[max_idx])
                                entities = [entities[max_idx]]
                                maximum_values.append(max_val)
                                print(f'#results: {num_entities}\ttoken: {token}\tentity to search: {entity}\tentities most similar: {entities}')
                            else:
                                more_entities = False
                        if len(maximum_values) > 0:
                            entities = [maximum_ents[np.argmax(maximum_values)]]
                # entities = list(self.lookup_table.get(token, max_entities)) if token in split_entity_batch else []
                # else:
                #     entities = list(self.lookup_table.get(token, []))[:max_entities]

                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_idx +=1
                    token_pos_idx = [pos_idx+1] # CHANGE
                    token_abs_idx = [abs_idx+1]
                    # token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)] # CHANGE
                    # token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    tokenized_entities = self.tokenizer.tokenize(ent)
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(tokenized_entities)+1)] # CHANGE
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(tokenized_entities) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    # add_word = list(word)
                    add_word = word
                    know_sent.append(add_word)
                    seg += [0]
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = self.tokenizer.tokenize(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1]
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num), dtype=int)
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = token_num

            # assert src_length == token_num
            if src_length < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

