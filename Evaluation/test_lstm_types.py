import sys

sys.path.append('../.')
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import Tokenizer.tokenizer as tokenizer
import random
import datetime
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


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


def get_data_from_file(path, ending):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.endswith(ending) or file.startswith("enc"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                #print('loading tokenized file: ', path)
                liste.append(f.read())

    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.endswith(ending) or file.startswith("enc"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    #print('loading tokenized file: ', file)
                    liste.append(f.read())

    return liste


def read_groundtruth(path):
    result = []
    file = os.path.basename(path)
    if file.endswith("txt"):
        with(open(path, "r", encoding="utf-8", errors="replace")) as f:
            result = f.readlines()
    return result


def get_examples(liste, tokenizer, type, ending, enc_type="normal"):
    result = []

    if enc_type == "normal":
        for i, file in enumerate(liste):
            context = tokenizer._tokenize(file.partition("[MASK]")[0])
            if type == "line" or type == "methodbody" or (type == "parameterlist" and ending == "json") or (type == "righthandAssignment" and ending == "json"):
                result.append(context[:len(context)-2])
            else:
                result.append(context)

    else:
        for i, file in enumerate(liste):
            if ending == "java":
                c = file.split("[ SEP ]")
                for enc in c[:len(c)-1]:
                    context = enc.partition("[ MASK ]")[0].split()
                    if type == "line" or type == "methodbody" or (type == "parameterlist" and ending == "json") or (type == "righthandAssignment" and ending == "json"):
                        result.append(context[:len(context)-2])
                    else:
                        result.append(context)
            else:
                c = file.split("[ S@@ E@@ P ]")
                for enc in c[:len(c)-1]:
                    context = enc.partition("[ M@@ A@@ S@@ K ]")[0].split()
                    if type == "line" or type == "methodbody" or (type == "parameterlist" and ending == "json") or (type == "righthandAssignment" and ending == "json"):
                        result.append(context[:len(context)-2])
                    else:
                        result.append(context)
    return result


def predict(device, net, words, gen_length):
    net.eval()
    new_text = []
    new_top5 = []
    (state_h, state_c) = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[w]]).to(device)
        out, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(F.softmax(out[0], dim=1), k=5)
    choice = top_ix.tolist()[0]
    new_text.append(choice[0])
    new_top5.append(choice)

    for _ in range(gen_length-1):
        ix = torch.tensor([choice]).to(device)
        out, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(F.softmax(out[0], dim=1), k=5)
        choice = top_ix.tolist()[0]
        new_text.append(choice[0])
        new_top5.append(choice)
    return new_text, new_top5


if __name__ == "__main__":
    in_path = "G:\\MASTER\\MODELS\\lstm\\java_enc\\best_lstm_java_enc.pth"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "lstm_java_enc_toked"

    #vocab_path = "G:\\MASTER\\raw_files\\Java\\small\\vocab_nltk.txt"
    #vocab_path = "G:\\MASTER\\raw_files\\CodeGru\\AST\\vocab\\vocab_merge.txt"
    #vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\bpe_ast\\useful\\vocab_bpe.txt"
    vocab_path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\bpe_java_vocab.txt"
    #vocab_path = "G:\\MASTER\\raw_files\\CodeGru\\java\\vocab_nltk.txt"
    #vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\vocab\\vocab.txt"

    # CHANGE FILE HERE
    orgs_path = "G:\\MASTER\\Evaluation\\enc_gt\\java\\enc_variables.txt"
    #orgs_path = "G:\\MASTER\\Evaluation\\enc_gt\\ast\\enc_variables.txt"
    #orgs_path = "G:\\MASTER\\Evaluation\\gt\\enc_variables.txt"

    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\variableidentifier\\done\\"
    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\variableidentifier\\ast\\"
    data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\typescen_enc\\java\\enc_java_var.txt"
    #data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\typescen_enc\\ast\\enc_ast_var.txt"
    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\CodeGru\\methodcalls\\ast\\"
    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\CodeGru\\methodcalls\\"

    type = "variableidentifier"
    enc_type = "enc"
    ending = "java"

    # normal gru
    seq_size = 32
    embed_size = 64
    lstm_size = 64
    dropout = 0.5

    max_samples = 50
    torch.manual_seed(66)
    random.seed(66)

    tokenizer = tokenizer.Tokenizer(vocab_path, "java")

    eval_files = get_data_from_file(data_path, ending)
    orgs = read_groundtruth(orgs_path)
    print("gts: ", len(orgs))
    tests = get_examples(eval_files, tokenizer, type, ending, enc_type)

    print("test examples: ", len(tests))

    model = RNNModule(tokenizer.get_vocab_len(), seq_size, embed_size, lstm_size, dropout)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model.load_state_dict(torch.load(in_path, map_location=device))
    # model = torch.load(in_path, map_location=device)
    model.to(device)
    criterion, optimizer = get_loss_and_train_op(model, 0.001)
    # predict
    model.eval()
    top_acc = 0
    acc = 0
    top_acc_t = 0
    acc_t = 0
    bleu = 0
    rogue = 0
    tok_count = 0

    dest = os.path.join(out_path, file_name + "_" +type+".txt")
    av = datetime.timedelta(0)
    for i, toks in enumerate(tests):
        toks = tokenizer.convert_tokens_to_ids(toks)
        start = datetime.datetime.now()
        ground_truth = orgs[i].rstrip("\n")
        toked_gt = tokenizer._tokenize(ground_truth)
        tok_count += len(toked_gt)

        top_1, top_5 = predict(device, model, toks, len(toked_gt))

        predictions = []
        for i, (t1, t5) in enumerate(zip(top_1, top_5)):
            id_gt = tokenizer.convert_tokens_to_ids(toked_gt[i])
            text_pred = tokenizer.convert_ids_to_tokens(t1)
            text_5 = tokenizer.convert_ids_to_tokens(t5)
            predictions.append(text_pred)
            if id_gt in t5:
                top_acc += 1
            if id_gt == t1:
                acc += 1
            if toked_gt[i] in text_5:
                top_acc_t += 1
            if toked_gt[i] == text_pred:
                acc_t += 1

        with open(dest, "a+", encoding='UTF-8') as f:
            try:
                f.write("{}\n".format(" ".join(x for x in predictions)))
            except TypeError as e:
                print("problem with: ", ground_truth)

    print("top5 accuracy", top_acc / max_samples)
    print("top1 accuracy", acc / max_samples)
    av = av / tok_count
    acc = acc / tok_count
    top_acc = top_acc / tok_count
    acc_t = acc_t / tok_count
    top_acc_t = top_acc_t / tok_count

    with open(dest, "a+") as f:
        f.write("--------\ntop1 accuracy: {}\ntop5 acc: {}\ntop1 text accuracy: {}\ntop5 text acc: {}\nno. of predictions: {}".format(
            acc, top_acc, acc_t, top_acc_t, tok_count))

