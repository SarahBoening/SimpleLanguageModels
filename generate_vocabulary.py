import os
from javalang import tokenizer as javatok
from itertools import chain

path_to_files = "E:\\IDEAProjects2\\"
dest = "E:\\IDEAProjects2\\"
directory = os.listdir(path_to_files)
files = []
tok = []
print("loading files...")
for file in directory:
    if file.endswith(".java"):
        source = os.path.join(path_to_files, file)
        with open(source, "r", errors='ignore') as f:
            text = f.read()
            files.append(text)
print("done, no of files: ", len(files))

print("tokenizing")
for t in files:
    split_tokens = [x.value for x in list(javatok.tokenize(t))]
    tok.append(split_tokens)

print("building vocabulary")
tok = list(chain.from_iterable(tok))
tok = list(dict.fromkeys(tok))
tok.sort()
dest = os.path.join(dest, "vocab.txt")

with open(dest, "w") as f:
    f.write("%s\n" % "[UNK]")
    for toks in tok:
        f.write("%s\n" % toks)

