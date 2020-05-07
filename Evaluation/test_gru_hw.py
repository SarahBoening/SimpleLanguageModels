import os
from itertools import chain

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tokenizer
import random
import datetime
from nltk.translate.bleu_score import sentence_bleu
import parse_nvidia_smi as gpuutil


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, gru_size, dropout):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.gru_size = gru_size
        self.drop = nn.Dropout(dropout)

        self.encode = nn.Embedding(n_vocab, embedding_size)
        self.gru = nn.GRU(embedding_size,
                          gru_size,
                          batch_first=True)
        self.decode = nn.Linear(gru_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.drop(self.encode(x))
        output, state = self.gru(embed, prev_state)
        logits = self.decode(output)
        preds = F.log_softmax(logits[0], dim=1)
        return logits, state

    def zero_state(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_size)


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
        if file.startswith("00") or file.startswith("enc"):
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
        # for l in liste:
        # text += l
        lines = liste[0].split(" ")
        for i in range(max_samples):
            rand = random.randint(4, len(lines) - 1)
            result.append(tokenizer.convert_tokens_to_ids(lines[rand - 4:rand + 4]))

    else:
        for line in liste:
            line = line.lstrip()
            if len(result) >= max_samples:
                break
            if line.startswith("/") or line.startswith("*") or line.startswith("import") or line.startswith(
                    "package") or line.startswith("@") or line.startswith("[") or "CLS" in line or "SEP" in line:
                pass
            else:
                tok_line = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(line))
                if 2 < len(tok_line) < 1024:
                    result.append(tok_line)

    return result


if __name__ == "__main__":
    pid = os.getpid()
    print(pid)
    proc = psutil.Process(pid)
    # UP NEXT: CODEGRU JAVA GLOBAL, ScenJava, ScenAst
    # UP NEXT: OPENVOCAB Java, AST, ScenAST, ScenJava (ENC)
    in_path = "G:\\MASTER\\MODELS\\gru\\ast_enc\\best_checkpoint-gru_ast_enc-1.0965386178338872.pth"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "gru_ast_enc_hw"
    # vocab_path = "G:\\MASTER\\raw_files\\CodeGru\\java\\vocab_nltk.txt"
    vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\bpe_ast\\useful\\vocab_bpe.txt"
    # data_path = "G:\\MASTER\\raw_files\\CodeGru\\java\\eval\\"
    data_path = "G:\\MASTER\\raw_files\\AST\\small\\bpe_ast\\useful\\temp\\"

    # normal gru
    seq_size = 32
    embed_size = 64
    gru_size = 64
    dropout = 0.5
    '''
    # codegru
    seq_size = 32
    embed_size = 300
    gru_size = 300
    dropout = 0.25
    '''
    max_samples = 100000
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
    if is_cuda:
        dev = "gpu"
    else:
        dev = "cpu"
    model.load_state_dict(torch.load(in_path, map_location=device))
    # model = torch.load(in_path, map_location=device)
    model.to(device)
    criterion, optimizer = get_loss_and_train_op(model, 0.001)
    # predict
    model.eval()
    cpu = 0.
    mem = 0.
    gpu = 0.
    dest = os.path.join(out_path, file_name + "_" + dev + ".txt")
    for i, toks in enumerate(tests):
        if i % 10000 == 0:
            print("progress: ", i, "/ ", max_samples)
        start = datetime.datetime.now()
        idx = random.randint(1, len(toks) - 1)
        ground_truth = toks[idx]
        tok_x = toks[:idx]
        state_h = model.zero_state(1)
        state_h.to(device)
        for t in tok_x:
            ix = torch.LongTensor([[t]]).to(device)
            out, state_h = model(ix, state_h)
        _, top_ix = torch.topk(F.softmax(out[0], dim=1), k=tokenizer.get_vocab_len())

        mem += proc.memory_info()[0] / 1e6
        cpu += proc.cpu_percent(None) / psutil.cpu_count()
        if is_cuda:
            gpu += gpuutil.get_gpu_util(0)


    mem = mem / max_samples
    cpu = cpu / max_samples
    gpu = gpu / max_samples
    print("Memory RSS: ", mem, " MB")
    print("CPU Percentage: ", cpu, "%")
    print("GPU Memory: ", gpu, " %")

    with open(dest, "a+") as f:
        f.write("Memory RSS: {} MB\nCPU Percentage: {}%\nGPU Memory: {}%\n".format(mem, cpu, gpu))
