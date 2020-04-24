import csv
import os
from nltk import trigrams
import nltk.corpus
from collections import defaultdict
import random
import datetime
import math
# based on: https://nlpforhackers.io/language-models/


def load_text(path):
    ''' loads all .raw files from path'''
    print("loading files...")
    text = ""
    list = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.startswith("token") and "valid" not in file or file.startswith("enc"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                list.append(f.read().split(" "))
                #text += f.read()
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if (file.startswith("enc_") or file.startswith("token")) and "valid" not in file:
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    list.append(f.read().split(" "))
                    #text += f.read()

    '''
    result = []
    for entry in list:
        for line in entry:
            line = line.rstrip("\n")
            line = line.rstrip("[CLS]")
            line = line.rstrip("[SEP]")
            l = len(line)
            if l > 0:
               result.append(line)
    '''
    print("done")
    return list


def save_ngram(model, output_path, n, corpus_name):
    ''' save model '''
    print('saving')
    file = os.path.join(output_path, "model_{}_{}.csv".format(n, corpus_name))
    with open(file, 'w+', encoding='UTF-8', errors='replace', newline='') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='\t')
        for w1_w2 in model:
            for w3 in model[w1_w2]:
                csvwriter.writerow([w1_w2, w3, model[w1_w2][w3]])
    print('done')


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


def predict(model, is_count, text, max_len):
    sentence_finished = False
    old = len(text)
    #text += [0]* max_len
    for i in range(max_len):
        # select a random probability threshold
        # random.seed(6)
        r = random.random()
        accumulator = .0
        if is_count:
            max_p = 0
            words = []
            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break
            '''
            for word in model[tuple(text[-2:])].keys():
                #print(word)
                p = model[tuple(text[-2:])][word]
                if p >= max_p:
                    words.append(word)
                    #text[i+old] = word
                #accumulator +=
                # select words that are above the probability threshold
                # if accumulator >= r:
            text.append(words[-1])
            print(words[-1])
            print(len(text))
            '''

        else:
            for word in model[str(tuple(text[-2:]))].keys():
                accumulator += model[str(tuple(text[-2:]))][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    #text.append(word)
                    break
        # if text[-2:] == [None, None]:
        # sentence_finished = True

    #print(' '.join([t for t in text if t]))
    return text


def model_ngram(n, data):
    print("modeling ngram ...")
    print(len(data))
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance
    for sentence in data:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    print("done")
    return model


def evaluate(model, text, iscount):
    total_loss= 0


def logscore(model, word, context):
    if model[context][word] == 0:
        return 0
    else:
        return math.log2(model[context][word])


def entropy(model, ngrams):
    x = [logscore(model, ngram[-1], ngram[:-1]) for ngram in ngrams]
    mean = sum(x)/len(x)
    return -1*mean


def perplexity(model, ngrams):
    return pow(2.0, entropy(model, ngrams))


if __name__ == "__main__":
    start = datetime.datetime.now()
    input_path = "/home/nilo4793/media/models/ngram/model_3_ast_small_n3.csv"
    output_path = "/home/nilo4793/media/models/ngram"
    eval_path = "/home/nilo4793/media/AST/smaller/raw_files/small/eval"
    data_path = "/home/nilo4793/media/AST/smaller/raw_files/small/train"
    #data_path = ""
    corpus = "ast_small_n3"
    gen = 10
    model = False
    load_data = False

    if load_data:
        if data_path:
            data = load_text(data_path)
        else:
            data = nltk.corpus.gutenberg.sents('austen-emma.txt')

    if not model:
        m = load_ngram(input_path)
        now = datetime.datetime.now()
        print("Time loading model: ", (now-start))

    else:
        n = 3
        m = model_ngram(n, data)
        now = datetime.datetime.now()
        print("modeling: ", now-start)
        save_ngram(m, output_path, n, corpus)

    now = datetime.datetime.now()
    print("with saving:", now-start)
    start = ["public", "static"]
    #start = [None, None]
    pred = predict(m, model, start, gen)
    print(pred)
    # evaluate
    eval = load_text(eval_path)
    for item in eval:
        test = trigrams(item, pad_right=True, pad_left=True)
    test_data = []
    for w in test:
        test_data.append(w)
    print("perplexity: ", perplexity(m, test_data))

