import os

path = "G:\\MASTER\\raw_files\\Java\\small\\bpe_java\\useful\\enc_java_eval_small.txt"

'''
files = os.listdir(path)
f_len = len(files)

for file in files:
    if file.startswith("tokenized"):
        print("fixing", file)
        source = os.path.join(path, file)
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            text = text.replace("[ CLS ]", "[CLS]")
            text = text.replace("[ SEP ]", "[UNK]")
            text = text.replace("[ UNK ]", "[SEP]")
        with open(source, 'w', encoding='utf-8', errors='replace') as f:
            f.write(text)
'''
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()
    text = text.replace("[ CLS ]", "[CLS]")
    text = text.replace("[ SEP ]", "[UNK]")
    text = text.replace("[ UNK ]", "[SEP]")
with open(path, 'w', encoding='utf-8', errors='replace') as f:
    f.write(text)
