import sys

sys.path.append('../.')
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from collections import Counter
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Tokenizer.tokenizer as tok
from itertools import chain
import datetime

parser = argparse.ArgumentParser(description='CodeGRU model')

parser.add_argument("--train_file", type=str, default="E:\\PyCharm Projects\\Master\\tokenized_0000.java_github_5k.raw",
                    help="input dir of data")
parser.add_argument("--eval_file", type=str, default="E:\\PyCharm Projects\\Master\\tokenized_0000.java_github_5k.raw",
                    help="input dir of eval data")
parser.add_argument("--output_name", type=str, default="gru_trump", help="Name of the model")
parser.add_argument("--checkpoint_path", type=str, default="./output/", help="output path for the model")
parser.add_argument("--vocab_path", type=str, default="E:\\PyCharm Projects\\Master\\vocab_nltk.txt",
                    help="path to vocab file")
parser.add_argument("--embedmodel_path", type=str, default="../Embedding/output/model.pth",
                    help="path to pretrained embedding model")
parser.add_argument("--ptmodel_path", type=str, default="", help="path to pretrained model")
parser.add_argument("--model_path", type=str, default="", help="path to trained model for eval or prediction")
parser.add_argument("--gpu_ids", type=int, default=0, help="IDs of GPUs to be used if available")
parser.add_argument("--epochs", type=int, default=10, help="No ofs epochs")
parser.add_argument("--seq_size", type=int, default=32, help="")
parser.add_argument("--batch_size", type=int, default=64, help="Size of batches")
parser.add_argument("--embedding_size", type=int, default=300, help="Embedding size for GRU network")
parser.add_argument("--gru_size", type=int, default=300, help="GRU size")
parser.add_argument("--dropout", type=float, default=0.25, help="GRU size")
parser.add_argument("--gradients_norm", type=int, default=5, help="Gradient normalization")
parser.add_argument("--initial_words", type=str, default="I, am",
                    help="string seperated by commas for list of initial words to predict further")
parser.add_argument("--do_predict", type=bool, default=False, help="should network predict at the end")
parser.add_argument("--do_train", type=bool, default=True, help="should network train")
parser.add_argument("--do_eval", type=bool, default=False, help="should network evaluate")
parser.add_argument("--do_finetune", type=bool, default=False, help="should network finetune, do_train has to be true")
parser.add_argument("--predict_top_k", type=int, default=5, help="Top k prediction")
parser.add_argument("--save_step", type=int, default=3000, help="steps to check loss and perpl")


def get_data_from_file(path, tokenizer):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        file = os.path.basename(path)
        if file.startswith("tokenized"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                liste.append(f.readlines())

    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("tokenized_"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    liste.append(f.readlines())

    print("done")
    in_text = []
    n_vocab = tokenizer.get_vocab_len()
    max_line = 0
    for file in liste:
        for line in file:
            line = line.rstrip("\n").split()
            l = len(line)
            if l > 0:
                in_text.append(tokenizer.convert_tokens_to_ids(line))
                if l >= max_line:
                    max_line = l
    print('Vocabulary size', n_vocab)
    return n_vocab, in_text, max_line


def get_zero_pad(line, i, max_len, batch_size):
    # returns context of variable-context sized learning
    zeros = (max_len-1) - (i+1)
    a = line[:i+1]
    b = line[1:i+2]
    max_y = int(np.ceil((len(a)+zeros) / batch_size))
    if max_y > 2:
        max_y = 2
    x = np.zeros((batch_size, max_y))
    y = np.zeros((batch_size, max_y))
    k = 0
    stop = False
    for i in range(x.shape[0]):
        if stop:
            break
        for j in range(x.shape[1]):
            if k >= len(a):
                stop = True
                break
            x[i, j] = a[k]
            y[i, j] = b[k]
            k += 1

    x = torch.tensor(x, dtype=torch.int64)
    y = torch.tensor(y, dtype=torch.int64)
    return x, y


def make_batches(int_text, args):
    print("building input and output vectors..")
    int_text = list(chain.from_iterable(int_text))
    num_batches = int(len(int_text) / (args.seq_size * args.batch_size))
    in_text = int_text[:num_batches * args.batch_size * args.seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (args.batch_size, -1))
    out_text = np.reshape(out_text, (args.batch_size, -1))
    print("done")
    print(in_text[:5])
    return in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i + seq_size], out_text[:, i:i + seq_size]


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, gru_size, dropout):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.gru_size = gru_size
        self.drop = nn.Dropout(dropout)

        self.encode = nn.Embedding(n_vocab, embedding_size)
        self.gru = nn.GRU(embedding_size,
                          gru_size,
                          batch_first=True)
        self.decode = nn.Linear(gru_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.drop(self.encode(x))
        output, state = self.gru(embed, prev_state)
        logits = self.decode(output)
        preds = F.log_softmax(logits[0], dim=1)
        return logits, state

    def zero_state(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_size)


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer


def predict(device, net, words, n_vocab, tokenizer, top_k=5):
    net.eval()
    #words = args.initial_words

    state_h = net.zero_state(1)
    state_h = state_h.to(device)
    for w in words:
        ix = torch.tensor([[tokenizer.convert_tokens_to_ids(w)]]).to(device)
        output, state_h = net(ix, state_h)

    choice = torch.argmax(output[0]).item()
    # _, top_ix = torch.topk(output[0], k=top_k)
    # choices = top_ix.tolist()
    # choice = choices[0][0]
    # print(tokenizer.convert_ids_to_tokens(choice))
    words.append(tokenizer.convert_ids_to_tokens(choice))

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, state_h = net(ix, state_h)

        # _, top_ix = torch.topk(output[0], k=top_k)
        # choices = top_ix.tolist()
        # choice = choices[0][0]
        choice = torch.argmax(output[0]).item()
        words.append(tokenizer.convert_ids_to_tokens(choice))
        # print(tokenizer.convert_ids_to_tokens(choice))
    print(' '.join(words).encode('utf-8'))


def evaluate(model, in_text, device, args, criterion):
    model.eval()
    total_loss = 0.
    state_h = model.zero_state(args.batch_size)
    state_h = state_h.to(device)
    # get data
    #batches = get_batches(in_text, out_text, args.batch_size, args.seq_size)
    total_loss = 0.
    nb_eval_steps = 0
    with torch.no_grad():
        for k, line in enumerate(in_text):
                max_len = len(line)
                for i in range(max_len-1):
                    x, y = get_zero_pad(line, i, max_len, args.batch_size)
                    x = x.to(device)
                    y = y.to(device)
                    logits, state_h = model(x, state_h)
                    loss = criterion(logits.transpose(1, 2), y)
                    loss_value = loss.item()
                    total_loss += loss_value
                    nb_eval_steps += 1
                if k % 1000 == 0:
                    l = total_loss / nb_eval_steps
                    print(k, ", ", math.exp(l))
    total_loss = total_loss / nb_eval_steps
    perplexity2 = math.exp(total_loss)
    return perplexity2



def main():
    args = parser.parse_args()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    torch.manual_seed(1)
    dev = 'cuda:' + str(args.gpu_ids)
    device = torch.device(dev if torch.cuda.is_available() else 'cpu')

    tokenizer = tok.Tokenizer(args.vocab_path, "java")
    if args.do_train:
        n_vocab, in_text, max_line = get_data_from_file(
            args.train_file, tokenizer)

        print("loading model and weights")
        net = RNNModule(n_vocab, args.seq_size,
                        args.embedding_size, args.gru_size, args.dropout)

        # load weights from embedding trained model
        #net.load_state_dict(torch.load(args.embedmodel_path, map_location=dev), strict=False)

        # freeze layers that do not need to be finetuned
        if args.do_finetune:
            print("loading pretrained model")
            net.load_state_dict(torch.load(args.ptmodel_path, map_location=dev))
            for name, param in net.named_parameters():
                if not name.startswith("gru"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        net = net.to(device)
        print("done")
        criterion, optimizer = get_loss_and_train_op(net, 0.001)
        best_ppl = 40.
        perpl = 40.
        iteration = 0
        total_loss = 0.
        start_time = datetime.datetime.now()
        plot_every = 50000
        reset_every = 200000
        all_losses = []
        j = 0
        ep_av = datetime.timedelta(0)
        line_av = datetime.timedelta(0)
        for e in range(args.epochs):
            ep_start = datetime.datetime.now()
            state_h = net.zero_state(args.batch_size)
            state_h = state_h.to(device)
            for k, line in enumerate(in_text):
                now = datetime.datetime.now()
                max_len = len(line)
                for i in range(max_len-1):
                    x, y = get_zero_pad(line, i, max_len, args.batch_size)
                    iteration += 1
                    j += 1
                    net.train()

                    optimizer.zero_grad()

                    x = x.to(device)
                    y = y.to(device)

                    logits, state_h = net(x, state_h)
                    loss = criterion(logits.transpose(1, 2), y)

                    loss_value = loss.item()
                    loss.backward()

                    state_h = state_h.detach()

                    _ = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), args.gradients_norm)

                    optimizer.step()

                    total_loss += loss_value
                    if j == reset_every:
                       best_ppl = 10.

                    if iteration % args.save_step == 0 and iteration > 0:
                        cur_loss = total_loss / args.save_step
                        perpl = math.exp(cur_loss)
                        elapsed = datetime.datetime.now() - start_time
                        print('Epoch: {}/{}'.format(e+1, args.epochs),
                              'Iteration: {}'.format(iteration),
                              'Loss: {}'.format(cur_loss),
                              'Perplexity: {}'.format(perpl),
                              'ms/batch: {}'.format(elapsed * 1000 / args.save_step))
                        total_loss = 0
                        start_time = datetime.datetime.now()

                        if perpl < best_ppl:
                            print("saving best checkpoint")
                            torch.save(net.state_dict(), os.path.join(args.checkpoint_path,
                                                                      'checkpoint_pt/best_checkpoint-{}-{}.pth'.format(
                                                                         args.output_name, perpl)))
                            best_ppl = perpl
                            j = 0

                    if iteration % plot_every == 0:
                        all_losses.append(total_loss / plot_every)
                        #total_loss = 0
                        #plt.figure()
                        #plt.plot(all_losses)
                        #plt.savefig(os.path.join(args.checkpoint_path, 'loss_plot_{}.png'.format(iteration)))
                        #plt.close()
<<<<<<< HEAD
=======
            
                line_av += (datetime.datetime.now() - now) / (k+1)

            print("line average: ", line_av)
            ep_av += (datetime.datetime.now() - ep_start) /(e+1)
            print("epoch average: ", ep_av)
>>>>>>> 79cf106e9cd93ef654f5a7c15d37821ff15df71f
        # save model after training
        torch.save(net, os.path.join(args.checkpoint_path, 'model-{}-{}.pth'.format(args.output_name, 'finished')))
        print('Finished training - perplexity: {}, loss: {}, best perplexity: {}'.format(perpl, total_loss, best_ppl))

        plt.figure()
        plt.plot(all_losses)
        plt.savefig(os.path.join(args.checkpoint_path, 'loss_plot.png'))

    else:
        print("loading model and weights")
        net = RNNModule(tokenizer.get_vocab_len(), args.seq_size,
                        args.embedding_size, args.gru_size, args.dropout)

        # load weights from embedding trained model
        net.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

        net = net.to(device)
        print("done")

    if args.do_eval:
        n_vocab, in_text, max_line = get_data_from_file(
            args.eval_file, tokenizer)
        #in_text, out_text = make_batches(in_text, args)
        criterion, optimizer = get_loss_and_train_op(net, 0.001)
        perpl = evaluate(net, in_text, device, args, criterion)
        print(perpl)
        file = os.path.join(args.checkpoint_path, args.output_name+"_eval.txt")
        with open(file, "w+") as f:
            f.write("perplexity: {}".format(perpl))

    if args.do_predict:
        words = args.initial_words.split(",")
        predict(device, net, words, n_vocab, tokenizer, top_k=5)


if __name__ == '__main__':
    main()
