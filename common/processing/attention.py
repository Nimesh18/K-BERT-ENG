import torch
from string import punctuation
from nltk.corpus import stopwords as nlstopwords
from transformers import RobertaTokenizerFast, RobertaModel

ROBERTA_PATH = 'sentence-transformers/nli-distilroberta-base-v2'

class SelfAttention(object):
    def __init__(self):
        self.model = RobertaModel.from_pretrained(ROBERTA_PATH, output_hidden_states=True)
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_PATH)
        self.special_tokens = list(set(self.tokenizer.special_tokens_map.values()))
        self.stopwords = nlstopwords.words("english") + list(punctuation) + self.special_tokens
        embed_dim = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    def get_entities_via_attention(self, sents):
        """
        input: dataset of sentences
        sentences will be of form `label\tsentence1\tsentence2`
        """
        sents1 = []
        sents2 = []
        for line in sents:
            line = line.strip().split('\t')
            sents1.append(line[1])
            sents2.append(line[2])
            
        encoding = self.tokenizer(sents1, text_pair=sents2, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            encoding = encoding.to(self.model.device)
            roberta_output = self.model(**encoding)

        embeddings = roberta_output.last_hidden_state
        tokens = [encoding.tokens(i) for i in range(len(sents))]
        important_words = self.get_important_words(embeddings, tokens)
        filtered_words = self.filter_important_words(important_words, self.stopwords)
        return filtered_words
        
    def get_important_words(self, embeddings, tokens):
        """
        embeddings shape: (L, N, S)
        L: target sequence length
        N: batch size
        S: source sequence length
        """
        query = embeddings.transpose(1,0).to('cpu')
        with torch.no_grad():
            attn_outputs, attn_weights = self.multihead_attn(query, query, query)

        indicies = self.sort_by_important_words(attn_weights)
        groupings = [self.roberta_relevant_token_map(token_set) for token_set in tokens]
        important_words = [self.group_relevant_tokens(group, words, ordering) for group, words, ordering in zip(groupings, tokens, indicies.tolist())]
        
        return important_words

    def sort_by_important_words(self, weights):
        """
        sort attention weights in descending order
        weights.shape is (N, L, S)
        N: batch size
        L: target sequence length
        S: source sequence length
        return indices with shape(N, L)
        """
        average = torch.mean(weights, dim=1)
        sorted_desc = torch.argsort(average, dim=1, descending=True)
        return sorted_desc



    def group_relevant_tokens(self, grouping, tokens, indicies):
        """
        group related tokens together
        """
        new_tokens = []
        while len(indicies) > 0:
            element = indicies.pop(0)
            if element in grouping:
                token_group = grouping[element]
                new_tokens.append(''.join(tokens[token_group[0]: token_group[-1] + 1]))
                for token_ind in token_group:
                    del grouping[token_ind]
        return new_tokens


    def roberta_relevant_token_map(self, tokens, prefix='Ġ'):
        ch = 0
        grouping = {}
        special_tokens = self.special_tokens
        avoid = special_tokens + list(punctuation)
        while ch < len(tokens):
            if tokens[ch] in avoid:
                grouping[ch] = [ch]
                ch+=1
                continue
            acc = [ch]
            while ch < len(tokens) - 1 and not tokens[ch + 1].startswith(prefix) and tokens[ch + 1] not in avoid:
                ch+=1
                acc.append(ch)
            for idx in acc:
                grouping[idx] = acc
            ch+=1
        return grouping

    def filter_important_words(self, important_words, stopwords):
        prefix_removed = list(map(self.remove_prefix, important_words))
        stopwords_removed = [[w for w in elements if w.lower() not in stopwords] for elements in prefix_removed]
        return stopwords_removed
            
    def remove_prefix(self, s, prefix='Ġ'):
        if type(s) is list:
            return [e.replace(prefix, '') for e in s]

        return s.replace(prefix, '')