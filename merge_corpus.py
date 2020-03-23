import os

path = "G:\\MASTER\\raw_files\\AST\\small\\train\\"
outpath = "G:\\MASTER\\raw_files\\AST\\small\\merge\\"

name = "merge_train"

files = os.listdir(path)
listfiles = []
i = 0
print(len(files))
for i in range(8, 13):
    if files[i].startswith("00"):
        source = os.path.join(path, files[i])
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            print("loading file ", files[i])
            listfiles.append(f.read())
dest = os.path.join(outpath, name + ".raw")

print("writing to file ...")
with open(dest, 'a+', encoding='utf-8', errors='replace') as f:
    f.writelines(' '.join(str(j) for j in i) + '\n' for i in listfiles)
