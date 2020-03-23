import os
from javalang import tokenizer as javatok
from itertools import chain
import nltk
import re
import matplotlib.pyplot as plt
from collections import Counter

#path_to_train_files="/home/nilo4793/media/Split_Corpus/smaller/large/train/"
path_to_train_files = "/home/nilo4793/media/AST/smaller/raw_files/small/eval/"
#path_to_train_files = "E:\\Hiwi\\BERT_undCo\\Split_Java_Corpus\\Java_split\\new_split\\train\\test\\"
path_to_eval_files = "/home/nilo4793/media/AST/smaller/raw_files/small/eval/"
#path_to_eval_files = "/home/nilo4793/media/Split_Corpus/smaller/large/eval/"
dest = "/home/nilo4793/media/AST/smaller/small/"
#dest = "/home/nilo4793/media/Split_Corpus/smaller/large/"
postfix = "onlyeval"
directory_t = os.listdir(path_to_train_files)
directory_e = os.listdir(path_to_eval_files)

files_t = []
tok = []
tok_split = []
tok_nl = []
print("loading training files...")
for file in directory_t:
    if file.endswith(".raw") and not file.startswith("cached") and not file.startswith("valid"):
        source = os.path.join(path_to_train_files, file)
        with open(source, "r", errors='replace', encoding='UTF-8') as f:
            text = f.read()
            print("tokenizing file ", file)
            #tok_split.append(text.split())
            tok_nl.append(nltk.word_tokenize(text))
print("done, no of files: ", len(tok_nl))
'''
files_e = []
print("loading eval files...")
for file in directory_e:
    if file.endswith(".raw") and not file.startswith("valid") and not file.startswith("cached"):
        source = os.path.join(path_to_eval_files, file)
        with open(source, "r", errors='replace', encoding='UTF-8') as f:
            text = f.read()
            tok_nl.append(nltk.word_tokenize(text))
print("done, no of files: ", len(tok_nl))
'''
#files = files_t + files_e
'''
print("tokenizing with javatokenizer")
for t in files:
    split_tokens = [x.value for x in list(javatok.tokenize(t))]
    tok.append(split_tokens)
tok = list(chain.from_iterable(tok))
count = Counter(tok)

print("tokenize by splitting")
for t in files:
    split_tokens = t.split()
    tok_split.append(split_tokens)

tok_split = list(chain.from_iterable(tok_split))
count_split = Counter(tok_split)
'''

print("tokenize by nltk")
#for t in files:
#    split_tokens = nltk.word_tokenize(t)
#    tok_nl.append(split_tokens)
tok_nl = list(chain.from_iterable(tok_nl))
tok_nl = re.findall(r"\w+|[^\w\s]", text)
count_nl = Counter(tok_nl)
del tok_nl
'''
#print("building vocabulary")
#countj = sorted(count.items(), key=lambda item: item[1], reverse=True)

#print("length before removing items < 5: ", len(countj))
#countj = {key:val for key, val in countj if val >= 5}
#print("length after removing items: ", len(countj))

dest_j = os.path.join(dest, "vocab_javatok.txt")
with open(dest_j, "w") as f:
    f.write("%s\n" % "[UNK]")
    for key in countj:
        f.write("%s\n" % key)

print("building vocabulary")
count_splitj = sorted(count_split.items(), key=lambda item: item[1], reverse=True)
print("length before removing items < 5: ", len(count_splitj))
count_splitj = {key:val for key, val in count_splitj if val >= 5}
print("length after removing items: ", len(count_splitj))

dest_s = os.path.join(dest, "vocab_split.txt")

with open(dest_s, "w") as f:
    f.write("{}\n{}\n{}\n".format("[UNK]","[CLS]", "[SEP]"))
    for key in count_splitj:
        f.write("%s\n" % key)
'''

print("building vocabulary")
count_nlj = sorted(count_nl.items(), key=lambda item: item[1], reverse=True)
del count_nl
print("length before removing items < 5: ", len(count_nlj))
count_nlj = {key:val for key, val in count_nlj if val >= 5}
print("length after removing items: ", len(count_nlj))

dest_s = os.path.join(dest, "vocab_nltk" + postfix + ".txt")

with open(dest_s, "w", encoding='UTF-8', errors='replace') as f:
    f.write("{}\n{}\n{}\n".format("[UNK]","[CLS]","[SEP]"))
    for key in count_nlj:
        f.write("%s\n" % key)

'''
# plot
plt.plot(sorted(count.values(), reverse=True))
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.savefig('Vocab_Freqs_Java.png')
plt.show()


# plot
plt.plot(sorted(count_split.values(), reverse=True))
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.savefig('Vocab_Freqs_Split.png')
plt.show()
'''
# plot
plt.plot(sorted(count_nlj.values(), reverse=True))
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.savefig(os.path.join(dest,'Vocab_Freqs_NLTK' + postfix + '.png'))
plt.show()
