# coding: utf-8
"""
KnowledgeGraph
"""
import numpy as np
import brain.config as config
from database import Database
from transformers import BertTokenizerFast
from common.embedding.transformer import RoBERTa
from common.utils import group_related_tokens, sym_matrix_to_vec


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, connurl, cache_path, cache_embedding_path, compute_embeddings=False):
        self.connurl = connurl
        self.lookup_table = Database(connurl, cache_path, cache_embedding_path if not compute_embeddings else None)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', never_split=config.NEVER_SPLIT_TAG)
        self.embedder = RoBERTa() if compute_embeddings else None
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def remove_trailing_padding(self, tokens, vm):
        if config.PAD_TOKEN not in tokens:
            return tokens, vm
        
        end = tokens.index(config.PAD_TOKEN)
        return tokens[:end], vm[:end,:end]


    def add_knowledge_with_vm(self, sent_batch, named_entities, seq=None, max_seq_len=None, max_length=128, threshold=0, e=None):
        """
        input: sent_batch - list of sentences, e.g., [["abcd", "abcd_pair"], ["efgh", "efgh_pair"]]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrices
        """
        split_sent_batch = [self.tokenizer(sent, text_pair=sent_pair) for sent, sent_pair in sent_batch]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        for sent, tokenized_sent in zip(sent_batch, split_sent_batch):

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            idx = -1
            ents_len = 0

            token_groups, entity_match_idxs, matches = group_related_tokens(sent, tokenized_sent, named_entities)

            for token_group_idx, token_group in enumerate(token_groups):
                entities = []
                idx += len(token_group)
                if token_group_idx in entity_match_idxs:
                    match_idx = entity_match_idxs.index(token_group_idx)
                    entity = matches[match_idx]
                    sequence = tokenized_sent.token_to_sequence(idx)
                    
                    concepts, concept_embeddings = self.lookup_table.retrieve_wiki_concepts(entity)
                    if len(concepts) > 0 and (seq is None or sequence == seq):

                        if self.embedder is not None:
                            instance_embeddings = self.embedder.get_sentence_embedding(list(map(lambda x: " ".join(x), concepts)))
                            sent_embedding = self.embedder.get_sentence_embedding(sent[sequence]).squeeze()
                        else:
                            sent_embedding = e[sent[sequence].strip()]
                            instance_embeddings = concept_embeddings

                        max_id, max_val = RoBERTa.get_most_similar(sent_embedding, instance_embeddings)

                        if max_val > threshold:
                            entities = concepts[max_id]
                            if max_seq_len is not None:
                                ents_len = ents_len + sum(list(map(lambda x: len(self.tokenizer.tokenize(x)), entities)))
                                if ents_len + len(tokenized_sent.tokens()) >= max_seq_len:
                                    entities = []


                sent_tree.append((token_group, entities))

                token_pos_idx = [pos_idx+i for i in range(1, len(token_group)+1)]
                token_abs_idx = [abs_idx+i for i in range(1, len(token_group)+1)]

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
            for i, (t_group, ents) in enumerate(sent_tree):

                know_sent += t_group

                pos += pos_idx_tree[i][0]
                for j, ent in enumerate(ents):
                    add_word = self.tokenizer.tokenize(ent)
                    know_sent += add_word
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num), dtype=np.uint8)
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
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                try:
                    sep_pos = know_sent.index(config.SEP_TOKEN)
                    len1 = sep_pos + 1
                    half = max_length // 2
                    if len1 <= half:
                        raise ValueError()
                    len2 = src_length - len1
                    if len2 <= half and len1 > half:
                        leftover = src_length - max_length
                        end1 = sep_pos - leftover
                        know_sent = know_sent[:end1] + know_sent[sep_pos:]
                        pos = pos[:end1] + pos[sep_pos:]

                        q1 = visible_matrix[:end1, :end1]
                        q2 = visible_matrix[sep_pos:, sep_pos:]
                        q3 = visible_matrix[:end1, sep_pos:]
                        q4 = visible_matrix[sep_pos:, :end1]
                        visible_matrix = np.hstack((np.vstack((q1, q4)), np.vstack((q3, q2))))
                    else:
                        m = np.math.ceil(max_length/2)
                        n = np.math.floor(max_length/2)
                        try:
                            sep_pos2 = know_sent.index(config.SEP_TOKEN, sep_pos + 1)
                        except ValueError as v1:
                            know_sent = know_sent[:n] + know_sent[sep_pos: sep_pos + m]
                            pos = pos[:n] + pos[sep_pos: sep_pos + m]

                            q1 = visible_matrix[:n, :n]
                            q2 = visible_matrix[sep_pos: sep_pos + m, sep_pos: sep_pos + m]
                            q3 = visible_matrix[:n, sep_pos: sep_pos + m]
                            q4 = visible_matrix[sep_pos: sep_pos + m, :n]
                            visible_matrix = np.hstack((np.vstack((q1, q4)), np.vstack((q3, q2))))
                        else:
                            know_sent = know_sent[:n] + know_sent[sep_pos: sep_pos + m - 1] + know_sent[sep_pos2: sep_pos2 + 1]
                            pos = pos[:n] + pos[sep_pos: sep_pos + m - 1] + pos[sep_pos2: sep_pos2 + 1]

                            q1 = visible_matrix[:n, :n]
                            q2 = visible_matrix[sep_pos: sep_pos + m - 1, sep_pos: sep_pos + m - 1]
                            q3 = visible_matrix[:n, sep_pos: sep_pos + m - 1]
                            q4 = visible_matrix[sep_pos: sep_pos + m - 1, :n]
                            q5 = visible_matrix[sep_pos2: sep_pos2 + 1, :n]
                            q6 = visible_matrix[sep_pos2: sep_pos2 + 1, sep_pos: sep_pos + m - 1]
                            q7 = visible_matrix[:n, sep_pos2: sep_pos2 + 1]
                            q8 = visible_matrix[sep_pos: sep_pos + m - 1, sep_pos2: sep_pos2 + 1]
                            q9 = visible_matrix[sep_pos2: sep_pos2 + 1, sep_pos2: sep_pos2 + 1]
                            # visible_matrix_sub = np.hstack((np.vstack((q1, q4, q5)), np.vstack((q3, q2, q6))))
                            visible_matrix = np.hstack((np.hstack((np.vstack((q1, q4, q5)), np.vstack((q3, q2, q6)))), np.vstack((q7, q8, q9))))


                except ValueError as v:
                    if know_sent[-1] != config.SEP_TOKEN:
                        know_sent = know_sent[:max_length]
                        pos = pos[:max_length]
                        visible_matrix = visible_matrix[:max_length, :max_length]
                    else:
                        know_sent = know_sent[:max_length - 1] + [know_sent[-1]]
                        pos = pos[:max_length - 1] + [pos[-1]]
                        q1 = visible_matrix[:max_length - 1, :max_length - 1]
                        q2 = visible_matrix[-1:, :max_length - 1]
                        q3 = visible_matrix[:max_length, -1:]
                        visible_matrix = np.hstack((np.vstack((q1, q2)), q3))
                    
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(sym_matrix_to_vec(visible_matrix))
        
        return know_sent_batch, position_batch, visible_matrix_batch