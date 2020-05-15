import copy
import os
import random
import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import LoadWatcher

def trim_tokens(tokenized_text, masked_index):
    if len(tokenized_text) > 512:
        if masked_index < 511:
            tokenized_text = tokenized_text[0:511]
        elif masked_index > len(tokenized_text) - 512:
            tokenized_text = tokenized_text[-512:]
        else:
            tokenized_text = tokenized_text[(masked_index - 256):(masked_index + 256)]

    return tokenized_text


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
            if file.startswith("00"):
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


def get_examples(liste, tokenizer, max_samples):
    result = []
    for line in liste:
        line = line.lstrip()
        if len(result) >= max_samples:
            break
        if line.startswith("/") or line.startswith("//") or line.startswith("*") or line.startswith("import") or line.startswith("package") or line.startswith("@") or line.startswith("[") or "CLS" in line or "SEP" in line:
            pass
        else:
            tok_line = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
            if 2 < len(tok_line) < 512:
                result.append(tok_line)

    return result


if __name__ == "__main__":
    pid = os.getpid()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    in_path = "G:\\MASTER\\MODELS\\BERT\\bert_ast_cp\\"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "bert_ast_glob"
    data_path = "G:\\MASTER\\raw_files\\AST\\small\\eval\\sicherungskopie\\"
    max_samples = 100000
    random.seed(66)
    torch.manual_seed(66)

    tokenizer = BertTokenizer.from_pretrained(in_path)

    eval_files = get_data_from_file(data_path, tokenizer)
    random.shuffle(eval_files)
    #max_samples = int(len(eval_files)*0.25)
    tests = get_examples(eval_files, tokenizer, max_samples)
    print(len(tests))
    model = BertForMaskedLM.from_pretrained(in_path)
    device = torch.device('cpu')
    model.to(device)

    # predict
    model.eval()
    top_acc = 0
    acc = 0
    mrr = 0

    av = datetime.timedelta(0)
    for i, toks in enumerate(tests):
        if i % 10000 == 0:
            print("progress: ", i, "/ ", max_samples)
        start = datetime.datetime.now()
        idx = random.randint(1, len(toks)-1)
        ground_truth = toks[idx]
        tok_x = copy.deepcopy(toks)
        tok_x[idx] = tokenizer.convert_tokens_to_ids('[MASK]')
        #tok_x = trim_tokens(tok_x, tok_x.index(ground_truth))
        segments_ids = [0]*len(tok_x)
        tokens_tensor = torch.tensor([tok_x]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, idx]).item()
            _, top_i = torch.topk(F.softmax(predictions[0, idx], dim=-1), k=tokenizer.vocab_size)
            choice = top_i.tolist()
        now = datetime.datetime.now()
        av += (now - start)
        if ground_truth in choice[:5]:
            #print("top 5")
            top_acc += 1
        if ground_truth == choice[0]:
            #print("top-1")
            acc += 1
        mrr += 1/(choice.index(ground_truth)+1)
        dest = os.path.join(out_path, file_name+".txt")
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
    with open(dest, "a") as f:
        f.write("top1 accuracy: {}\ntop5 acc: {}\nno. of samples: {}\nMRR: {}\naverage prediction time: {}".format(acc/max_samples, top_acc/max_samples, max_samples, mrr, av))
