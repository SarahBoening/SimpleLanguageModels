import copy
import os
import random
import datetime
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil

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
    in_path = "G:\\MASTER\\MODELS\\GPT-2\\best_gpt2_small_ast\\"
    out_path = "G:\\MASTER\\Evaluation\\"
    file_name = "gpt2_ast_glob"
    data_path = "G:\\MASTER\\raw_files\\AST\\small\\eval\\sicherungskopie\\"
    length = 1
    max_samples = 100000
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
    device = torch.device('cpu')
    model.to(device)

    # predict
    model.eval()
    top_acc = 0
    acc = 0
    mrr = 0
    av = datetime.timedelta(0)
    for i, toks in enumerate(tests):
        if i % 10000 == 0:
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
        now = datetime.datetime.now()
        av += (now - start)
        if ground_truth in choice[:5]:
            #print("top 5")
            top_acc += 1
        if ground_truth == choice[0]:
            #print("top-1")
            acc += 1
        mrr += 1/(choice.index(ground_truth)+1)
        dest = os.path.join(out_path, file_name+".txt")
        with open(dest, "a+", encoding='UTF-8') as f:
            try:
                l = tokenizer.decode(toks)
                top = tokenizer.decode(choice[:5])
                f.write("{}\tgt: {}\ttop5: {}\n".format(l, tokenizer.decode(toks[idx]), top))
            except TypeError as e:
                print("problem with: ", choice)
    print("top5 accuracy", top_acc/max_samples)
    print("top1 accuracy", acc/max_samples)
    av = av/max_samples
    mrr = mrr/max_samples
    with open(dest, "a+", encoding='UTF-8') as f:
        f.write("top1 accuracy: {}\ntop5 acc: {}\nno. of samples: {}\nMRR: {}\naverage prediction time: {}".format(acc/max_samples, top_acc/max_samples, max_samples, mrr, av))

