import sys

sys.path.append('../.')
import copy
import os
from collections import defaultdict
from itertools import chain
import random
import datetime
import Tokenizer.tokenizer as tokenizer
from nltk import trigrams

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
                    context = enc.partition("[ MASK ]")[0]
                    context = context.split()
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


def load_ngram(input_path):
    '''load model'''
    print('loading model')
    model = defaultdict(dict)
    with open(input_path, 'r', encoding='UTF-8', errors='replace') as csv_file:
        try:
            for line in csv_file:
                w1_w2, w3, prob = line.split('\t')
                model[str(w1_w2)][str(w3)] = float(prob)
        except ValueError as e:
            print("Error with line: ", line)
    print('done')
    return model

def predict(model, text, max_len):
    sentence_finished = False
    old = len(text)
    #text += [0]* max_len
    result = [[] for i in range(max_len)]
    for i in range(max_len):
        # select a random probability threshold
        # random.seed(6)
        r = random.random()
        accumulator = .0
        pred = []
        words = []
        for word in model[str(tuple(text[-2:]))].keys():
            p = model[str(tuple(text[-2:]))][word]
            pred.append(p)
            words.append(word)
        preds = copy.deepcopy(pred)
        preds.sort(reverse=True)
        for j in preds:
            idx = pred.index(j)
            x = words[idx]
            result[i].append(x)
        if(len(result[i]) > 0):
            text.append(result[i][0])
        else:
            text.append(" ")
    return result

if __name__ == "__main__":
    in_path = "G:\\MASTER\\MODELS\\ngrams\\model_3_java_enc_2_n3.csv"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "ngram_java_enc_toked"

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
    max_samples = 50
    random.seed(66)

    tokenizer = tokenizer.Tokenizer(vocab_path, "java")

    eval_files = get_data_from_file(data_path, ending)
    orgs = read_groundtruth(orgs_path)
    print("gts: ", len(orgs))
    tests = get_examples(eval_files, tokenizer, type, ending, enc_type)

    print("test examples: ", len(tests))

    model = load_ngram(in_path)
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
        start = datetime.datetime.now()
        ground_truth = orgs[i].rstrip("\n")
        toked_gt = tokenizer._tokenize(ground_truth)
        tok_count += len(toked_gt)
        a = toks[-2]
        b = toks[-1]

        choice = predict(model, [a, b], len(toked_gt))
        predictions = []
        for i, t1 in enumerate(choice):
            if len(t1) > 0:
                predictions.append(t1[0])
                if toked_gt[i] in t1[:5]:
                    # print("top 5")
                    top_acc += 1
                if len(t1) >= 1 and toked_gt[i] == t1[0]:
                    # print("top-1")
                    acc += 1

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

