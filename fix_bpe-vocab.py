import os


filepath = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\enc_java_train_small.txt"
vocab = {}
with open(filepath, encoding='utf-8', errors='replace') as f:
    toks = f.read().split(" ")

for t in toks:
    if t not in vocab.keys():
        vocab[t] = 1
    else:
        vocab[t] += 1

dest = os.path.join("G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\", "bpe_vocab_java_train.txt")

with open(dest, 'w+', encoding='utf-8', errors='replace') as f:
    for key, val in vocab.items():
        if int(val) >= 5:
            f.write("%s\n" % key)
