from javalang import tokenizer as javatok
import collections
import re
import nltk
from itertools import chain
'''
Based on the tokenizer from Huggingface Transformer
'''

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Tokenizer:
    def __init__(self, vocab_file, type):
        super(Tokenizer, self).__init__()
        if vocab_file:
            self.vocab = load_vocab(vocab_file)
        else:
            raise ValueError("No vocabulary given")
        self.path = vocab_file
        self.unk_token = "[UNK]"
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.type = type

    def _tokenize(self, text):
        if self.type == "java":
            """tokenizes a text with the java tokenizer"""
            data = nltk.word_tokenize(text)
            for i, word in enumerate(data):
                data[i] = re.findall(r"\w+|[^\w\s]", data[i], re.UNICODE)
            return list(chain.from_iterable(data))
            #return [x.value for x in list(javatok.tokenize(text))]
        else:
            return []

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.
            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def get_vocab_len(self):
        return len(self.ids_to_tokens)
