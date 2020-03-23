import os
import random
import shutil
import glob


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def make_corpus(folders, outpath):
    corpus_output_path_500 = os.path.join(outpath, "small")
    if not os.path.isdir(corpus_output_path_500):
        os.mkdir(corpus_output_path_500)
    corpus_output_path_1 = os.path.join(outpath, "medium")
    if not os.path.isdir(corpus_output_path_1):
        os.mkdir(corpus_output_path_1)
    corpus_output_path_2 = os.path.join(outpath, "large")
    if not os.path.isdir(corpus_output_path_2):
        os.mkdir(corpus_output_path_2)
    not_full_500 = True
    not_full_1 = True
    not_full_2 = True
    for i, folder in enumerate(folders):
        try:
            list = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(folder)] for val in sublist]
            for file in list:
                if file.endswith(".java") and os.path.isfile(file):
                    source = os.path.join(folder, file)
                    base = os.path.basename(file)
                    if not_full_500:
                        file_path = os.path.join(corpus_output_path_500, base)
                        shutil.copy(source, file_path)
                    if not_full_1:
                        file_path = os.path.join(corpus_output_path_1, base)
                        shutil.copy(source, file_path)
                    if not_full_2:
                        file_path = os.path.join(corpus_output_path_2, base)
                        shutil.copy(source, file_path)

                size5 = get_directory_size(corpus_output_path_500)
                size1 = get_directory_size(corpus_output_path_1)
                size2 = get_directory_size(corpus_output_path_2)
                if not_full_500 and size5 >= int(5e8):
                    print("No. of projects: ", i)
                    not_full_500 = False
                if not_full_1 and size1 >= int(1e9):
                    print("No. of projects: ", i)
                    not_full_1 = False
                if not_full_2 and size2 >= int(2e9):
                    print("No. of projects: ", i)
                    not_full_2 = False
        except FileNotFoundError:
            print("Folder not found: ", folder)
        except PermissionError:
            print("Permission denied for file: ", file)

def make_eval(folders, outpath, n_projects):
    ''' select projects for evaluation not in train corpus'''
    file = os.path.join(outpath, 'projects_{}.txt'.format(str(n_projects)))
    with open(file, 'w') as f:
        for i, j in enumerate(reversed(folders)):
            if i >= n_projects:
                return
            f.write(j+"\n")

path = "H:\\Master\\Corpus\\Java\\java_projects\\java_projects\\"
path = "/home/nilo4793/media/java_projects/"

output_path = "H:\\Master\\Corpus\\Java\\smaller\\"
output_path = "/home/nilo4793/media/Split_Corpus/small_corp/"

size = "small" # small = 500MB medium = 1 GB large = 2GB

folders = [f.path for f in os.scandir(path) if f.is_dir()]

random.seed(66)
random.shuffle(folders)

print("Building the corpus")
make_corpus(folders, output_path)
make_eval(folders, output_path, 10)

