import sys

sys.path.append('../.')
import copy
import os
import random
import datetime
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil
import parse_nvidia_smi as gpuutil

def get_data_from_file(path, tokenizer):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.startswith("00"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                liste.append(f.readlines())

    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("00"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    liste.append(f.readlines())
    result = []
    for entry in liste:
        for line in entry:
            line = line.rstrip("\n")
            l = len(line)
            if l > 0:
               result.append(line)

    return result


def get_examples(liste, tokenizer, max_samples):
    result = []
    for i in range(100):
        if len(result) >= max_samples:
            break
        for line in liste:
            line = line.lstrip()
            if len(result) >= max_samples:
                break
            if line.startswith("/") or line.startswith("*") or line.startswith("import") or line.startswith(
                    "package") or line.startswith("@") or line.startswith("[") or "CLS" in line or "SEP" in line:
                pass
            else:
                tok_line = tokenizer.encode(line)
                if 2 < len(tok_line) < 1024:
                    result.append(tok_line)

    return result


if __name__ == "__main__":
    pid = os.getpid()
    print(pid)
    proc = psutil.Process(pid)

    in_path = "/home/nilo4793/media/models/gpt2/gpt2_scenario/_gpt2_scenario"
    out_path = "/home/nilo4793/media/Evaluation/"
    file_name = "gpt2_java_scen_hw"
    data_path = "/home/nilo4793/media/scenario/java/eval"
    length = 1
    max_samples = 10000
    random.seed(66)
    torch.manual_seed(66)

    tokenizer = GPT2Tokenizer.from_pretrained(in_path)

    eval_files = get_data_from_file(data_path, tokenizer)
    random.shuffle(eval_files)
    # max_samples = int(len(eval_files)*0.25)
    #max_samples = 100
    tests = get_examples(eval_files, tokenizer, max_samples)
    print(len(tests))
    model = GPT2LMHeadModel.from_pretrained(in_path)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    if is_cuda:
        dev = "gpu"
    else:
        dev = "cpu"
    model.to(device)

    # predict
    model.eval()
    top_acc = 0
    cpu = 0.
    mem = 0.
    gpu = 0.
    dest = os.path.join(out_path, file_name + "_" + dev + ".txt")
    for i, toks in enumerate(tests):
        if i % 1000 == 0:
            print("progress: ", i, "/ ", max_samples)
        start = datetime.datetime.now()
        idx = random.randint(1, len(toks) - 1)
        ground_truth = toks[idx]
        tok_x = copy.deepcopy(toks)
        tok_x = tok_x[:idx]
        encoded_prompt = torch.tensor([tok_x]).to(device)
        past = None
        with torch.no_grad():
            outputs = model(encoded_prompt)
            next_token_logits = outputs[0][0, -1, :]
        _, top_idx = torch.topk(next_token_logits, k=tokenizer.vocab_size)
        choice = top_idx.tolist()
        mem += proc.memory_info()[0] / 1e6
        cpu += proc.cpu_percent(None) / psutil.cpu_count()
        if is_cuda:
            gpu += gpuutil.get_gpu_util(0)

    mem = mem / max_samples
    cpu = cpu / max_samples
    gpu = gpu / max_samples
    print("Memory RSS: ", mem, " MB")
    print("CPU Percentage: ", cpu, "%")
    print("GPU Memory: ", gpu, " %")
    with open(dest, "a") as f:
        f.write("Memory RSS: {} MB\nCPU Percentage: {}%\nGPU Memory: {}%\n".format(mem, cpu, gpu))

