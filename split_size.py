import os
import shutil


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


path = "/home/nilo4793/media/Split_Corpus/train"
outpath_2 = "/home/nilo4793/media/Split_Corpus/2GB"
outpath_500 = "/home/nilo4793/media/Split_Corpus/500MB"
outpath_1 = "/home/nilo4793/media/Split_Corpus/1GB"


files = os.listdir(path)

for f in files:
    if get_directory_size(outpath_500) <= int(5e8):
        file_path = os.path.join(outpath_500, f)
        source = os.path.join(path,  os.path.basename(f))
        shutil.copy(source, file_path)
    if get_directory_size(outpath_1) <= int(1e9):
        file_path = os.path.join(outpath_1, f)
        source = os.path.join(path, os.path.basename(f))
        shutil.copy(source, file_path)
    if get_directory_size(outpath_2) <= int(2e9):
        file_path = os.path.join(outpath_2, f)
        source = os.path.join(path, os.path.basename(f))
        shutil.copy(source, file_path)
