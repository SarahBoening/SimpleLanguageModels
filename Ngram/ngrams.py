import csv
import os
from nltk import ngrams, trigrams
import nltk.corpus
from collections import Counter, defaultdict
import random
import dill as pickle


# based on: https://nlpforhackers.io/language-models/


def load_text(path):
    ''' loads all .raw files from path'''
    print("loading files...")
    text = ""
    list = []
    if os.path.isfile(path):
        if path.startswith("tokenized_"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                text = f.read()
                #list.append(text)
        elif not path.startswith('cached') and path.endswith(".raw") and not os.path.isfile("tokenized_" + path):
            print('loading and tokenizing file: ', path)
            with open(path, "r", encoding="utf-8", errors='replace') as f:
                text = tokenizer._tokenize(f.read())
                list.append(text)
                dest = "tokenized_" + path
                with open(dest, "w", encoding="utf-8", errors="replace")as f:
                    f.write(' '.join(text))
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("tokenized_"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    text += ' '.join(f.read().split())
                    #list.append(text)

            elif not file.startswith('cached') and file.endswith(".raw") and not os.path.isfile(
                    os.path.join(path, "tokenized_" + file)):
                print('loading and tokenizing file: ', file)

                with open(source, "r", encoding="utf-8", errors='replace') as f:
                    text += tokenizer._tokenize(f.read())
                    #list.append(text)
                    dest = os.path.join(path, "tokenized_" + file)
                    with open(dest, "w", encoding="utf-8", errors="replace")as f:
                        f.write(' '.join(text))
    print("done")
    return text


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

    for i in range(max_len):
        # select a random probability threshold
        # random.seed(6)
        r = random.random()
        accumulator = .0
        if is_count:
            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break
        else:
            for word in model[str(tuple(text[-2:]))].keys():
                accumulator += model[str(tuple(text[-2:]))][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break
        # if text[-2:] == [None, None]:
        # sentence_finished = True

    print(' '.join([t for t in text if t]))


def evaluate(model):
    # TODO
    pass


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


if __name__ == "__main__":
    input_path = ""
    output_path = "./output/"
    data_path = "/home/nilo4793/Documents/Thesis/corpora/Java/small/train/"
    # data_path = ""
    corpus = "java_small_n3"
    gen = 5
    model = True
    load_data = True

    if load_data:
        if data_path:
            data = load_text(data_path)
        else:
            data = nltk.corpus.gutenberg.sents('austen-emma.txt')

    if not model:
        m = load_ngram(input_path)
    else:
        n = 3
        m = model_ngram(n, data)
        save_ngram(m, output_path, n, corpus)

    start = [None, None]
    predict(m, model, start, gen)
