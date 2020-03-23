import sys
import datetime

sys.path.append('../.')

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizer as tok
from itertools import chain
import math

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.encode = nn.Embedding(vocab_size, embedding_dim)  # embeddings
        self.decode = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.encode(inputs)
        add_embeds = torch.sum(embeds, dim=0).view(1, -1)
        out = self.decode(add_embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def load_text(path, tokenizer):
    ''' loads all .raw files from path'''
    print("loading files...")
    text = ""
    list = []
    if os.path.isfile(path):
        if path.startswith("tokenized_"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                text = f.read()
                list.append(text)
        elif not path.startswith('cached') and path.endswith(".raw") and not os.path.isfile("tokenized_" + path):
            print('loading and tokenizing file: ', path)
            with open(path, "r", encoding="utf-8", errors='replace') as f:
                text = tokenizer._tokenize(f.read())
                list.append(text)
                dest = "tokenized_" + path
                with open(dest, "w", encoding="utf-8", errors="replace")as f:
                    f.write(' '.join(text))
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("tokenized_"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    text = f.read().split()
                    list.append(text)

            elif not file.startswith('cached') and file.endswith(".raw") and not os.path.isfile(
                    os.path.join(path, "tokenized_" + file)):
                print('loading and tokenizing file: ', file)

                with open(source, "r", encoding="utf-8", errors='replace') as f:
                    text = tokenizer._tokenize(f.read())
                    list.append(text)
                    dest = os.path.join(path, "tokenized_" + file)
                    with open(dest, "w", encoding="utf-8", errors="replace")as f:
                        f.write(' '.join(text))
    print("done")
    return list


torch.manual_seed(1)

#path = "G:\\MASTER\\raw_files\\AST\\small\\eval\\"
path = "/home/nilo4793/raid/corpora/AST/small/train/"
outpath = "/home/nilo4793/raid/output/embedding/javasmall/"
#outpath = "G:\\MASTER\\outputs\\embeddings\\"

#vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\vocab.txt"
vocab_path = "/home/nilo4793/raid/corpora/AST/small/vocab.txt"

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
tokenizer = tok.Tokenizer(vocab_path, "java")

raw_text = load_text(path, tokenizer)