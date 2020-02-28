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
    if os.path.isfile(path):
        with open(path, "r", errors='ignore') as f:
            text += f.read()
    else:
        files = os.listdir(path)
        for file in files:
            if file.endswith(".raw"):
                source = os.path.join(path, file)
                with open(source, "r", errors='ignore') as f:
                    text += f.read()
                    #list.append(text)
    print("done")
    return text


def preprocess_text(data_path):
    data = nltk.tokenize.sent_tokenize(load_text(data_path))
    for i, sent in enumerate(data):
        data[i] = sent.split()
    return data


def save_ngram(model, output_path, n, corpus_name):
    ''' save model '''
    file = os.path.join(output_path, "model_{}_{}.pkl".format(n, corpus_name))
    output = open(file, 'wb')
    pickle.dump(model, output)
    output.close()


def load_ngram(input_path):
    '''load model'''
    pkl_file = open(input_path, 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model


def predict(model, text, max_len):
    sentence_finished = False

    for i in range(max_len):
        # select a random probability threshold
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        #if text[-2:] == [None, None]:
            #sentence_finished = True

    print(' '.join([t for t in text if t]))


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
    input_path = "./Ngram/output/model_3_trump.pkl"
    output_path = "./Ngram/output/"
    data_path = "./LSTM/data/trump.txt"
    #data_path = ""
    corpus = "trump"
    gen = 200
    model = True
    if data_path:
        data = preprocess_text(data_path)
    else:
        data = nltk.corpus.gutenberg.sents('austen-emma.txt')

    if not model:
        m = load_ngram(input_path)
    else:
        n = 3
        m = model_ngram(n, data)
        save_ngram(m, output_path, n, corpus)

    start = [None, None]
    predict(m, start, gen)
