import sys

sys.path.append('../.')

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Tokenizer.tokenizer as tok
from itertools import chain


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
                text = f.read()
                list.append(tokenizer._tokenize(text))
                dest = "tokenized_" + path
                with open(dest, "w", encoding="utf-8", errors="replace")as f:
                    f.writelines(' '.join(str(j) for j in i) + '\n' for i in list)
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("tokenized_"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    text = f.read()
                    list.append(text)

            elif not file.startswith('cached') and file.endswith(".raw") and not os.path.isfile(
                    os.path.join(path, "tokenized_" + file)):
                print('loading and tokenizing file: ', file)

                with open(source, "r", encoding="utf-8", errors='replace') as f:
                    text = f.read()
                    list.append(tokenizer._tokenize(text))
                    dest = os.path.join(path, "tokenized_" + file)
                    with open(dest, "w", encoding="utf-8", errors="replace")as f:
                        f.writelines(' '.join(str(j) for j in i) + '\n' for i in list)
    print("done")
    return list


torch.manual_seed(1)

# path = "0000.java_github_5k.raw"
path = "/home/nilo4793/raid/corpora/Java/small/train/"
outpath = "/home/nilo4793/raid/output/embedding/javasmall/"
# outpath = "G:\\MASTER\\outputs\\embeddings\\"

# vocab_path = "vocab_small.txt"
vocab_path = "/home/nilo4793/raid/corpora/Java/small/vocab_nltk.txt"
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
tokenizer = tok.Tokenizer(vocab_path, "java")

raw_text = list(chain.from_iterable(load_text(path, tokenizer)))

# By deriving a set from `raw_text`, we deduplicate the array
vocab_size = tokenizer.get_vocab_len()
print("building context vectors")
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print("done")
print(data[:5])
# make_context_vector(data[0][0], word_to_ix)  # example

# loss model optimizer
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, embedding_dim=64)
optimizer = optim.SGD(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print("starting training")
# 10 epoch
best_ppl = 100000
perpl = 100000
oldloss = 10000000
epochs = 100
iteration = 0
for epoch in range(epochs):
    print("Epoch ", epoch, "/ ", epochs)
    total_loss = torch.FloatTensor([0])
    for context, target in data:
        iteration += 1
        model.train()
        context_idxs = [tokenizer.convert_tokens_to_ids(w) for w in context]
        target_idx = tokenizer.convert_tokens_to_ids(target)
        context_var = Variable(torch.LongTensor(context_idxs))
        target_var = Variable(torch.LongTensor([target_idx]))
        model.zero_grad()
        log_probs = model(context_var)
        winner = tokenizer._convert_id_to_token(torch.argmax(log_probs[0]).item())
        loss = loss_function(log_probs, target_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if iteration % 100 == 0 and iteration > 0:
            cur_loss = total_loss / 100.
            perpl = math.exp(cur_loss)
            elapsed = time.time() - start_time
            print('Epoch: {}/{}'.format(e, args.epochs),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(cur_loss),
                  'Perplexity: {}'.format(perpl),
                  'ms/batch: {}'.format(elapsed * 1000 / 100))
            total_loss = 0
            start_time = time.time()
        if perpl < best_ppl:
            print("saving best checkpoint")
            torch.save(net.state_dict(), os.path.join(args.checkpoint_path,
                                                      'checkpoint_pt/best_checkpoint-{}-{}.pth'.format(args.output_name,
                                                                                                       perpl)))
            best_ppl = perpl

        losses.append(total_loss)

torch.save(model.state_dict(), os.path.join(outpath, "cbow_finished_loss_{}.pth".format(total_loss)))
