import copy
import os
import random
import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import LoadWatcher


def trim_tokens(tokenized_text, masked_index):
    if len(tokenized_text) > 1024:
        if masked_index < 1023:
            tokenized_text = tokenized_text[0:1023]
        elif masked_index > len(tokenized_text) - 1024:
            tokenized_text = tokenized_text[-1024:]
        else:
            tokenized_text = tokenized_text[(masked_index - 512):(masked_index + 512)]

    return tokenized_text


def get_data_from_file(path, ending):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.endswith(ending):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                # print('loading tokenized file: ', path)
                liste.append(f.read())

    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.endswith(ending):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    # print('loading tokenized file: ', file)
                    liste.append(f.read())

    return liste

def read_groundtruth(path, ending):
    result = []
    file = os.path.basename(path)
    if file.endswith("txt"):
        with(open(path, "r", encoding="utf-8", errors="replace")) as f:
            #print('loading tokenized file: ', path)
            result = f.readlines()
    return result


def get_examples(liste, tokenizer, type="normal"):
    result = []
    if type == "normal":
        for i, file in enumerate(liste):
            context = tokenizer.encode(file.partition("[MASK]")[0])
            result.append(context)

    else:
        for i, file in enumerate(liste):
            context = tokenizer.encode(file.partition("[MASK]")[0])
            result.append(context)


    return result


if __name__ == "__main__":
    in_path = "G:\\MASTER\\MODELS\\GPT-2\\best_gpt_small_java\\"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "gpt2_java_concat"

    # CHANGE FILE HERE
    #orgs_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\line\\originals\\"
    orgs_path = "G:\\MASTER\\Evaluation\\gt\\Variables.txt"

    data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\variableidentifier\\done\\"
    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\methodcalls\\ast\\"
    # data_path = "G:\\MASTER\\Small_Corp\\small\\Pred_categories\\CodeGru\\variableidentifier\\done\\"
    type = "variables"
    enc_type = "normal"
    ending = "java"
    max_samples = 50
    random.seed(66)
    torch.manual_seed(66)

    tokenizer = GPT2Tokenizer.from_pretrained(in_path)

    eval_files = get_data_from_file(data_path, ending)
    orgs = read_groundtruth(orgs_path, ending)

    # max_samples = int(len(eval_files)*0.25)
    tests = get_examples(eval_files, tokenizer, enc_type)
    print(len(tests))
    model = GPT2LMHeadModel.from_pretrained(in_path)
    device = torch.device('cpu')
    model.to(device)

    # predict
    model.eval()
    top_acc = 0
    acc = 0
    top_acc_t = 0
    acc_t = 0
    bleu = 0
    rogue = 0
    # bleu_cand = []
    # bleu_ref = []
    dest = os.path.join(out_path, file_name + "_" + type + ".txt")
    av = datetime.timedelta(0)
    mask_idx = tokenizer.encode("[MASK]")
    for i, toks in enumerate(tests):
        top1 = []
        toks.append(mask_idx)
        # toks = tokenizer.convert_tokens_to_ids(toks)
        start = datetime.datetime.now()
        ground_truth = orgs[i].rstrip("\n")
        tokenized_target = tokenizer.encode(ground_truth)
        for i in range(len(tokenized_target)):
        # print(tokenizer.tokenize(target)
            s_target = tokenized_target[i]
            idx = toks.index(mask_idx)
            toks = trim_tokens(toks, idx)
            idx = toks.index(mask_idx)

            tok_x = copy.deepcopy(toks)
            tok_x = tok_x[:idx]
            # bleu_ref.append([[ground_truth]])
            tokens_tensor = torch.tensor([tok_x]).to(device)
            with torch.no_grad():
                outputs = model(tokens_tensor)
                next_token_logits = outputs[0][0, -1, :]
                _, top_idx = torch.topk(next_token_logits, k=tokenizer.vocab_size)
                choice = top_idx.tolist()
                choice_text = tokenizer.decode(choice)
                top1.append(choice[0])

                toks[idx] = choice[0]
                toks.append(mask_idx)


            # bleu_cand.append([choice_text[0]])
            now = datetime.datetime.now()
            av += (now - start)
            if s_target in choice[:5]:
                # print("top 5")
                top_acc += 1
            if s_target == choice[0]:
                # print("top-1")
                acc += 1

        with open(dest, "a+", encoding='UTF-8') as f:
            try:
                top = tokenizer.decode(top1)
                f.write("gt: {}\ttoked gt: {}\ttop1: {}\n".format(ground_truth, tokenizer.decode(tokenized_target), top))
            except TypeError as e:
                print("problem with: ", choice)

    print("top5 accuracy", top_acc / max_samples)
    print("top1 accuracy", acc / max_samples)
    av = av / max_samples
    acc = acc / max_samples
    top_acc = top_acc / max_samples
    # bleu = corpus_bleu(bleu_ref, bleu_cand)
    # print(bleu)
    # rogue = rogue / max_samples

    with open(dest, "a+") as f:
        f.write(
            "top1 accuracy: {}\ntop5 acc: {}\ntop1 text accuracy: {}\ntop5 text acc: {}\nno. of samples: {}\nBLEU: {}\nROUGE: {}\naverage prediction time: {}".format(
                acc, top_acc, acc_t, top_acc_t, max_samples, bleu, rogue, av))
