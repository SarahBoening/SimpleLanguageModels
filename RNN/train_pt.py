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

parser = argparse.ArgumentParser(description='Baseline GRU model')

parser.add_argument("--train_file", type=str, default="/home/nilo4793/Documents/Thesis/BPE/java_0125_text_enc.txt",
                    help="input dir of data")
parser.add_argument("--output_name", type=str, default="lstm_BPE_Test", help="Name of the model")
parser.add_argument("--checkpoint_path", type=str, default="./output/", help="output path for the model")
parser.add_argument("--vocab_path", type=str, default="./vocab.txt", help="path to vocab file")
parser.add_argument("--embedmodel_path", type=str, default="../Embedding/output/model.pth",
                    help="path to pretrained embedding model")
parser.add_argument("--gpu_ids", type=int, default=0, help="IDs of GPUs to be used if available")
parser.add_argument("--epochs", type=int, default=10, help="No ofs epochs")
parser.add_argument("--seq_size", type=int, default=32, help="")
parser.add_argument("--batch_size", type=int, default=16, help="Size of batches")
parser.add_argument("--embedding_size", type=int, default=64, help="Embedding size for GRU network")
parser.add_argument("--lstm_size", type=int, default=64, help="GRU size")
parser.add_argument("--dropout", type=float, default=0.5, help="GRU size")
parser.add_argument("--gradients_norm", type=int, default=5, help="Gradient normalization")
parser.add_argument("--initial_words", type=list, default=['public', 'class'],
                    help="List of initial words to predict further")
parser.add_argument("--do_predict", type=boolean, default=True, help="should network predict at the end")
parser.add_argument("--predict_top_k", type=int, default=5, help="Top k prediction")
parser.add_argument("--save_step", type=int, default=1000, help="steps to check loss and perpl")


def get_data_from_file(path, batch_size, seq_size, tokenizer):
    print("loading files...")
    text = ""
    liste = []
    if os.path.isfile(path):
        if path.startswith("tokenized_"):
            with(open(path, "r", encoding="utf-8", errors="replace")) as f:
                print('loading tokenized file: ', path)
                text = f.read().split()
                liste.append(text)
        elif not path.startswith('cached') and path.endswith(".raw") and not os.path.isfile("tokenized_" + path):
            print('loading and tokenizing file: ', path)
            with open(path, "r", encoding="utf-8", errors='replace') as f:
                text = tokenizer._tokenize(f.read())
                liste.append(text)
                dest = "tokenized_" + path
                with open(dest, "w", encoding="utf-8", errors="replace")as f:
                    f.write(' '.join(text))
    else:
        files = os.listdir(path)
        for file in files:
            source = os.path.join(path, file)
            if file.startswith("tokenized_"):
                with(open(source, "r", encoding="utf-8", errors="replace")) as f:
                    print('loading tokenized file: ', file)
                    text = f.read().split()
                    liste.append(text)

            elif not file.startswith('cached') and file.endswith(".raw") and not os.path.isfile(
                    os.path.join(path, "tokenized_" + file)):
                print('loading and tokenizing file: ', file)

                with open(source, "r", encoding="utf-8", errors='replace') as f:
                    text = tokenizer._tokenize(f.read())
                    liste.append(text)
                    dest = os.path.join(path, "tokenized_" + file)
                    with open(dest, "w", encoding="utf-8", errors="replace")as f:
                        f.write(' '.join(text))
    print("done")

    n_vocab = tokenizer.get_vocab_len()

    print('Vocabulary size', n_vocab)
    raw_text = list(chain.from_iterable(liste))
    int_text = tokenizer.convert_tokens_to_ids(raw_text)
    print("building batches..")
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    print("done")
    return n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i + seq_size], out_text[:, i:i + seq_size]


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size, dropout):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.drop = nn.Dropout(dropout)
        self.encode = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.decode = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.drop(self.encode(x))
        output, state = self.lstm(embed, prev_state)
        logits = self.decode(output)
        preds = F.log_softmax(logits, dim=1)
        return preds, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer


def predict(device, net, words, n_vocab, tokenizer, top_k=5):
    net.eval()
    words = args.initial_words

    state_h = net.zero_state(1)
    state_h = state_h.to(device)
    for w in words:
        ix = torch.tensor([[tokenizer.converts_token_to_id(w)]]).to(device)
        output, state_h = net(ix, state_h)

    choice = torch.argmax(output[0]).item()
    # _, top_ix = torch.topk(output[0], k=top_k)
    # choices = top_ix.tolist()
    # choice = choices[0][0]
    print(tokenizer.convert_ids_to_tokens(choice))
    words.append(tokenizer.convert_ids_to_tokens(choice))

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, state_h = net(ix, state_h)

        # _, top_ix = torch.topk(output[0], k=top_k)
        # choices = top_ix.tolist()
        # choice = choices[0][0]
        choice = torch.argmax(output[0]).item()
        words.append(tokenizer.convert_ids_to_tokens(choice))
        print(tokenizer.convert_ids_to_tokens(choice))
    print(' '.join(words).encode('utf-8'))


def evaluate(model, tokenizer, criterion):
    # TODO write evaluation
    model.eval()
    total_loss = 0.
    state_h, state_c = net.zero_state(args.batch_size)
    # get data
    with torch.no_grad():
        # iterate over batches
        # data = input, y = target
        logits, (state_h, state_c) = model(data, (state_h, state_c))
        total_loss += len(data) * criterion(logits.transpose(1, 2), y).item()
    return total_loss / (len(data_source) - 1)


def main():
    args = parser.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    dev = 'cuda:' + str(args.gpu_ids)
    device = torch.device(dev if torch.cuda.is_available() else 'cpu')

    tokenizer = tok.Tokenizer(args.vocab_path, "java")

    n_vocab, in_text, out_text = get_data_from_file(
        args.train_file, args.batch_size, args.seq_size, tokenizer)
    print("loading model and weights")
    net = RNNModule(n_vocab, args.seq_size,
                    args.embedding_size, args.lstm_size, args.dropout)

    # load weights from embedding trained model
    # TODO test if is working
    net.load_state_dict(torch.load(args.embedmodel_path), strict=False)

    net = net.to(device)
    print("done")
    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0
    total_loss = 0.
    start_time = datetime.datetime.now()
    best_ppl = 10
    perpl = 10
    plot_every = 25000
    all_losses = []
    for e in range(args.epochs):
        batches = get_batches(in_text, out_text, args.batch_size, args.seq_size)
        state_h, state_c = net.zero_state(args.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            net.train()

            optimizer.zero_grad()

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            loss_value = loss.item()

            loss.backward()

            state_h = state_h.detach()
            state_c = state_c.detach()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), args.gradients_norm)

            optimizer.step()

        total_loss += loss_value

        if iteration % args.save_step == 0 and iteration > 0:
            cur_loss = total_loss / args.save_step
            perpl = math.exp(cur_loss)
            elapsed = datetime.datetime.now() - start_time
            print('Epoch: {}/{}'.format(e, args.epochs),
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
                                                              args.output_name,
                                                              perpl)))
                best_ppl = perpl

        if iteration % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
            plt.figure()
            plt.plot(all_losses)
            plt.savefig(os.path.join(args.checkpoint_path, 'loss_plot_{}.png', format(iteration)))

    # save model after training
    torch.save(net, os.path.join(args.checkpoint_path, 'model-{}-{}.pth'.format(args.output_name, 'finished')))
    print('Finished training - perplexity: {}, loss: {}, best perplexity: {}'.format(perpl, total_loss, best_ppl))

    plt.figure()
    plt.plot(all_losses)
    plt.savefig(os.path.join(args.checkpoint_path, 'loss_plot.png'))

    if args.do_predict:
        predict(device, net, args.initial_words, n_vocab, tokenizer, top_k=5)


if __name__ == '__main__':
    main()
