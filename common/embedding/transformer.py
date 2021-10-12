import re
import torch
from transformers import RobertaTokenizerFast, RobertaModel, CharSpan
from sentence_transformers import util


ROBERTA_PATH = 'sentence-transformers/nli-distilroberta-base-v2'

class RoBERTa(object):
    def __init__(self):
        self.model = RobertaModel.from_pretrained(ROBERTA_PATH, output_hidden_states=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_PATH)


    def get_embedding(self, token_id, charspans, sent, layers = list(range(-4, 0))):
        """
        get embedding for word corresponding to token_id using RoBERTa Tokenizer
        """
        encoding = self.tokenizer(sent, return_tensors='pt')
        with torch.no_grad():
            encoding = encoding.to(self.model.device)
            roberta_output = self.model(**encoding)

        word_idxs = self.get_word_index_range(token_id, charspans, sent)
        word_embedding = self.get_word_embedding(roberta_output.hidden_states, encoding.attention_mask, word_idxs, layers)
        return word_embedding

    def get_sentence_embedding(self, sents):
        encoding = self.tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            encoding = encoding.to(self.model.device)
            roberta_output = self.model(**encoding)

        return self.mean_pooling(roberta_output, encoding['attention_mask'])

    def get_word_index_range(self, token_id, charspans, sent):
        word_ids, mapping = self.get_char_map(sent)
        word_start = mapping[charspans[token_id].start]
        word = word_ids[word_start]
        # word_end = mapping[word.end - 1]
        for t in range(token_id, len(charspans)):
            if charspans[t].end == word.end:
                return token_id, t
            
        return token_id, token_id


    def get_char_map(self, sent):
        """
        split sent into words
        create character to word map
        return word_idxs, mapping
        """
        mapping = {}
        word_ids = []
        idx = 0
        for word in self.split_words(sent, strip=False):
            length = len(word.strip())
            if length == 0:
                idx +=1
                continue
            start = idx
            span = CharSpan(start, start + length)
            word_ids.append(span)
            for ch in word:
                mapping[idx] = len(word_ids) - 1
                idx +=1
            # idx +=1
        return word_ids, mapping


    @staticmethod
    def get_character_spans(encoded_input):
        return [encoded_input.token_to_chars(tok) for tok in
            [idx for idx, e in enumerate(encoded_input.word_ids()) if e is not None]]


    def split_words(self, sentence, trim=True, strip=True):
        """
        split sentence into words
        """
        words = re.split(r'([\W+])', sentence)
        if trim:
            return list(filter(lambda x: len(x.strip() if strip else x) > 0, words))
        return words


    def group_by_sequence_ids(self, sequence_ids):
        """
        group by sequence id
        """
        token_sequence_map = {}
        for seq_idx, seq_id in enumerate(sequence_ids):
            if seq_id is not None:
                if seq_id not in token_sequence_map:
                    token_sequence_map[seq_id] = [seq_idx]
                else:
                    token_sequence_map[seq_id].append(seq_idx)
        return token_sequence_map


    def get_word_embedding(self, hidden_states, attention_mask, word_idxs, layers):
        """
        return mean of token embedding across the last layers
        """
        word_idx_s, word_idx_e = word_idxs
        if 0 in attention_mask[word_idx_s: word_idx_e + 1]:
            raise Exception("Word index corresponds to 0 in attention mask")

        embedding = torch.stack([hidden_states[layer][:, word_idx_s: word_idx_e + 1].mean(1) for layer in layers]).mean(0).squeeze()
        return embedding[0] if embedding.dim() > 1 else embedding


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings [0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def get_most_similar(word_embedding, sentence_embeddings, n=1):
        """
        get the top n similar sentence embeddings to word_embedding
        """
        similarities = [util.pytorch_cos_sim(word_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
        max_idx = 0
        maximum = 0
        for idx, similarity in enumerate(similarities):
            if similarity > maximum:
                maximum = similarity
                max_idx = idx

        return max_idx, maximum