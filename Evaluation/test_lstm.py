import os

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tokenizer
import random
import datetime
import LoadWatcher
import parse_nvidia_smi as gpuutil

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size, dropout):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.drop = nn.Dropout(dropout)

        self.encode = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                          lstm_size,
                          batch_first=True)
        self.decode = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.drop(self.encode(x))
        output, state = self.lstm(embed, prev_state)
        logits = self.decode(output)
        preds = F.log_softmax(logits[0], dim=1)
        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer

def get_data_from_file(path, tokenizer):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.startswith("00"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                liste.append(f.readlines())

    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("00") or file.startswith("enc"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    liste.append(f.readlines())
    result = []
    for entry in liste:
        for line in entry:
            line = line.rstrip("\n")
            l = len(line)
            if l > 0:
               result.append(line)

    return result

def get_examples(liste, tokenizer, max_samples, type="normal"):
    result = []
    new = []
    done = False
    print(type)
    if type == "enc":
        text = ""
        #for l in liste:
            #text += l
        lines = liste[0].split(" ")
        for i in range(max_samples):
            rand = random.randint(4, len(lines)-1)
            result.append(tokenizer.convert_tokens_to_ids(lines[rand-4:rand+4]))

    else:
        for line in liste:
            line = line.lstrip()
            if len(result) >= max_samples:
                break
            if line.startswith("/") or line.startswith("*") or line.startswith("import") or line.startswith("package") or line.startswith("@") or line.startswith("[") or "CLS" in line or "SEP" in line:
                pass
            else:
                tok_line = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(line))
                if 2 < len(tok_line) < 1024:
                    result.append(tok_line)

    return result

if __name__ == "__main__":
    # UP NEXT: GRU EncJava, EncAst
    in_path = "G:\\MASTER\\MODELS\\lstm\\java_enc\\best_lstm_java_enc.pth"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "lstm_java_hw"
    vocab_path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\bpe_java_vocab.txt"
    #vocab_path = "G:\\MASTER\\raw_files\\Java\\small\\vocab_nltk.txt"
    data_path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\temp\\"
    #data_path = "G:\\MASTER\\raw_files\\Java\\small\\eval\\"
    seq_size = 32
    embed_size = 64
    gru_size = 64
    dropout = 0.5
    max_samples = 10000
    torch.manual_seed(66)
    random.seed(66)

    tokenizer = tokenizer.Tokenizer(vocab_path, "java")

    eval_files = get_data_from_file(data_path, tokenizer)
    random.shuffle(eval_files)

    tests = get_examples(eval_files, tokenizer, max_samples, "enc")
    print(len(tests))
    model = RNNModule(tokenizer.get_vocab_len(), seq_size, embed_size, gru_size, dropout)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model.load_state_dict(torch.load(in_path, map_location=device))
    #model = torch.load(in_path, map_location=device)
    model.to(device)
    criterion, optimizer = get_loss_and_train_op(model, 0.001)

    # predict
    model.eval()
    top_acc = 0
    acc = 0
    mrr = 0
    av = datetime.timedelta(0)
    av = datetime.timedelta(0)
    for i, toks in enumerate(tests):
        if i % 10000 == 0:
            print("progress: ", i, "/ ", max_samples)
        start = datetime.datetime.now()
        idx = random.randint(1, len(toks)-1)
        ground_truth = toks[idx]
        tok_x = toks[:idx]
        state_h, state_c = model.zero_state(1)
        state_h.to(device)
        state_c.to(device)
        for t in tok_x:
            ix = torch.LongTensor([[t]]).to(device)
            out, (state_h, state_c) = model(ix, (state_h, state_c))
        _, top_ix = torch.topk(F.softmax(out[0], dim=1), k=tokenizer.get_vocab_len())
        choice = top_ix.tolist()[0]
        now = datetime.datetime.now()
        #av += (now - start)
        if ground_truth in choice[:5]:
            #print("top 5")
            top_acc += 1
        if ground_truth == choice[0]:
            #print("top-1")
            acc += 1
        mrr += 1/(choice.index(ground_truth)+1)

        dest = os.path.join(out_path, file_name + ".txt")

        with open(dest, "a+", encoding='UTF-8') as f:
            try:
                l = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(toks))
                top = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(choice[:5]))
                f.write("{}\tgt: {}\ttop5: {}\n".format(l, tokenizer.convert_ids_to_tokens(toks[idx]), top))
            except TypeError as e:
                print("problem with: ", choice)

    print("top5 accuracy", top_acc/max_samples)
    print("top1 accuracy", acc/max_samples)
    av = av/max_samples
    mrr = mrr/max_samples
    with open(dest, "a+") as f:
        f.write("top1 accuracy: {}\ntop5 acc: {}\nno. of samples: {}\nMRR: {}\naverage prediction time: {}".format(acc/max_samples, top_acc/max_samples, max_samples, mrr, av))
