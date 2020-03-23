import os
from itertools import chain

inpath = "G:\\MASTER\\raw_files\\Java\\small\\vocab\\"
outpath = "G:\\MASTER\\raw_files\\Java\\small\\vocab\\"

files = os.listdir(inpath)

toks = []

for file in files:
    print("loading file ", file)
    source = os.path.join(inpath, file)
    with open(source, encoding='utf-8', errors='replace') as f:
        toks.append([line.rstrip() for line in f])

toks = list(chain.from_iterable(toks))
tok = list(dict.fromkeys(toks))

print(len(tok))

dest = os.path.join(outpath, "vocab.txt")
with open(dest, 'w', encoding='utf-8', errors='replace') as f:
    for t in toks:
        f.write("%s\n" % t)
