import copy
import os
from collections import defaultdict
from itertools import chain
import random
import datetime
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
                #text += f.read()
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if (file.startswith("enc_") or file.startswith("tok")) and file.endswith("raw") and "valid" not in file:
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    list.append(f.readlines())
                    #text += f.read()
    result = []
    for entry in list:
        for line in entry:
            line = line.rstrip("\n")
            #lines = line.split(";")
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
    #text += [0]* max_len
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
    in_path = "G:\\MASTER\\MODELS\\ngrams\\model_3_ast_small_n3.csv"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "ngram_ast_glob_test"
    #data_path = "G:\\MASTER\\raw_files\\AST\\small\\bpe_ast\\useful\\enc_ast_eval_small.txt"
    data_path = "G:\\MASTER\\raw_files\\AST\\small\\eval\\"
    #vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\vocab\\vocab.txt"
    vocab_path = "G:\\MASTER\\raw_files\\AST\\small\\vocab\\vocab.txt"
    max_samples = 50
    random.seed(66)

    tokenizer = tokenizer.Tokenizer(vocab_path, "java")
    eval_files = load_text(data_path)

    random.shuffle(eval_files)
    #max_samples = int(len(eval_files)*0.25)
    tests = get_examples(eval_files, max_samples, tokenizer)
    print(len(tests))
    load_start = datetime.datetime.now()
    model = load_ngram(in_path)
    load_now = datetime.datetime.now()
    print("time loading model: ", (load_now - load_start))
    # predict
    top_acc = 0
    acc = 0
    mrr = 0
    dest = os.path.join(out_path, file_name+".txt")

    av = datetime.timedelta(0)
    for i, line in enumerate(tests):
        if i % 10000 == 0:
            print("progress: ", i, "/ ", max_samples)
        start = datetime.datetime.now()
        test = []
        x = trigrams(line)
        for i in x:
            test.append(i)
        random.shuffle(test)
        a,b, ground_truth = test[0]
        choice = predict(model, [a, b], 1)
        now = datetime.datetime.now()
        av += (now - start)
        if ground_truth in choice[:5]:
            #print("top 5")
            top_acc += 1
        if ground_truth in choice:
            mrr += 1/(choice.index(ground_truth)+1)
        if len(choice)>= 1 and ground_truth == choice[0]:
            #print("top-1")
            acc += 1

        with open(dest, "a+", encoding='UTF-8') as f:
            try:
                l = tokenizer.convert_tokens_to_string(line)
                top = ' '.join(z for z in choice[:5])
                top1 = ""
                if len(choice)>= 1:
                    top1 = choice[0]
                f.write("{}\tgt: {}\ttop5: {}\n".format(l, top1, top))
            except TypeError as e:
                print("problem with: ", choice)

    print("top5 accuracy", top_acc/max_samples)
    print("top1 accuracy", acc/max_samples)
    av = av/max_samples
    mrr = mrr/max_samples
    with open(dest, "a+") as f:
        f.write("top1 accuracy: {}\ntop5 acc: {}\nno. of samples: {}\nMRR: {}\naverage prediction time: {}".format(acc/max_samples, top_acc/max_samples, max_samples, mrr, av))
