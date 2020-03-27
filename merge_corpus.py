import os

path = "/home/nilo4793/media/Split_Corpus/smaller/small/eval/"
outpath = "/home/nilo4793/media/Split_Corpus/smaller/small/merge/"

name = "merge_toked_eval"

files = os.listdir(path)
listfiles = []
i = 0
size = len(files)
print(size)
for i in range(0, size):
    if files[i].startswith("tokenized"):
        source = os.path.join(path, files[i])
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            print("loading file ", files[i])
            listfiles.append(f.read())
dest = os.path.join(outpath, name + ".raw")

print("writing to file ...")
with open(dest, 'w', encoding='utf-8', errors='replace') as f:
    for l in listfiles:
        f.write("{}\n".format(l))
