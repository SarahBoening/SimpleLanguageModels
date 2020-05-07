import copy
import os
from collections import defaultdict
from itertools import chain
import random
import datetime

import psutil
import tokenizer
from nltk import trigrams


def load_ngram(input_path):
    '''load model'''
    print('loading model')
    model = defaultdict(dict)
    with open(input_path, 'r', encoding='UTF-8', errors='replace') as csv_file:
        for line in csv_file:
            w1_w2, w3, prob = line.split('\t')
            model[str(w1_w2)][str(w3)] = float(prob)
    print('done')
    return model


def load_text(path):
    ''' loads all .raw files from path'''
    print("loading files...")
    text = ""
    list = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.startswith("tok") or file.startswith("enc") and "valid" not in file:
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                list.append(f.readlines())
                # text += f.read()
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if (file.startswith("enc_") or file.startswith("00")) and (file.endswith("raw") or file.endswith("txt")) and "valid" not in file:
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    list.append(f.readlines())
                    # text += f.read()
    result = []
    for entry in list:
        for line in entry:
            line = line.rstrip("\n")
            # lines = line.split(";")
            if len(line) > 0:
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


def predict(model, text, max_len):
    sentence_finished = False
    old = len(text)
    # text += [0]* max_len
    for i in range(max_len):
        # select a random probability threshold
        # random.seed(6)
        r = random.random()
        accumulator = .0
        pred = []
        words = []
        result = []
        for word in model[str(tuple(text[-2:]))].keys():
            p = model[str(tuple(text[-2:]))][word]
            pred.append(p)
            words.append(word)
        preds = copy.deepcopy(pred)
        preds.sort(reverse=True)
        for i in preds:
            idx = pred.index(i)
            result.append(words[idx])
    return result


if __name__ == "__main__":
    pid = os.getpid()
    print(pid)
    proc = psutil.Process(pid)
    in_path = "G:\\MASTER\\MODELS\\ngrams\\model_3_ast_small_n3.csv"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "ngram_ast_hw"
    #data_path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\temp\\"
    data_path = "G:\\MASTER\\raw_files\\AST\\small\\eval\\"
    #vocab_path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\bpe_java_vocab.txt"
    vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\vocab\\vocab.txt"
    max_samples = 10000
    random.seed(66)
    dev = "cpu"
    tokenizer = tokenizer.Tokenizer(vocab_path, "java")
    eval_files = load_text(data_path)

    random.shuffle(eval_files)

    tests = get_examples(eval_files, tokenizer, max_samples, "enc")
    print(len(tests))
    load_start = datetime.datetime.now()
    model = load_ngram(in_path)
    load_now = datetime.datetime.now()
    print("time loading model: ", (load_now - load_start))
    # predict
    cpu = 0.
    mem = 0.
    gpu = 0.
    dest = os.path.join(out_path, file_name + "_" + dev + ".txt")

    for i, line in enumerate(tests):
        if i % 10000 == 0:
            print("progress: ", i, "/ ", max_samples)
        test = []
        x = trigrams(line)
        for i in x:
            test.append(i)
        random.shuffle(test)
        a, b, ground_truth = test[0]
        choice = predict(model, [a, b], 1)
        mem += proc.memory_info()[0] / 1e6
        cpu += proc.cpu_percent(None) / psutil.cpu_count()

    mem = mem / max_samples
    cpu = cpu / max_samples
    gpu = gpu / max_samples
    print("Memory RSS: ", mem, " MB")
    print("CPU Percentage: ", cpu, "%")
    print("GPU Memory: ", gpu, " %")

    with open(dest, "a+") as f:
        f.write("Memory RSS: {} MB\nCPU Percentage: {}%\nGPU Memory: {}%\n".format(mem, cpu, gpu))
