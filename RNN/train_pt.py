import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
from argparse import Namespace
import time

import model as m

# https://github.com/pytorch/examples/blob/master/word_language_model/

flags = Namespace(
    train_file='./data/trump.txt',
    output_name='gru_trump',
    checkpoint="",  # name of checkpoint to reload
    out_file="",  # name for file with generated text
    do_train=True,
    seed=666,
    epochs=200,
    seq_size=32,
    batch_size=32,
    embedding_size=64,  # size of word embeddings
    lstm_size=64,
    nhid=200,
    nlayers=2,
    gradients_norm=5,
    initial_words=['I', 'am'],
    words=100,  # length of new generated words
    do_predict=True,
    load_checkpoint=False,
    predict_top_k=5,
    checkpoint_path='./output/',
    rnn_type='GRU',  # 'GRU' oder 'LSTM'
    lr=0.001
)


def get_data_from_file(train_file, batch_size, seq_size):
    # TODO for several files
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i + seq_size], out_text[:, i:i + seq_size]


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def predict(device, model, n_vocab, vocab_to_int, int_to_vocab, top_k):
    model.eval()
    words = flags.initial_words
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for w in words:
            ix = torch.tensor([[vocab_to_int[w]]]).to(device)
            output, hidden = model(ix, hidden)

        choice = torch.argmax(output[0]).item()
        #_, top_ix = torch.topk(output[0], k=top_k)
        #choices = top_ix.tolist()
        #choice = choices[0][0]
        #print(int_to_vocab[choice])
        words.append(int_to_vocab[choice])
        for _ in range(flags.words):
            ix = torch.tensor([[choice]]).to(device)
            output, hidden = model(ix, hidden)

            #_, top_ix = torch.topk(output[0], k=top_k)
            #choices = top_ix.tolist()
            #choice = choices[0][0]
            choice = torch.argmax(output[0]).item()
            words.append(int_to_vocab[choice])
            print(int_to_vocab[choice])
    print(' '.join(words).encode('utf-8'))


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train(batches, model, device, optimizer, criterion, n_vocab, total_loss, iteration, e):
    loss = 0
    model.train()
    hidden = model.init_hidden(flags.batch_size)

    for x, y in batches:
        iteration += 1

        model.zero_grad()
        hidden = repackage_hidden(hidden)

        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        output, hidden = model(x, hidden)

        loss = criterion(output.transpose(1,2), y)
        loss_value = loss.item()
        loss.backward()
        total_loss += loss.item()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        _ = torch.nn.utils.clip_grad_norm_(
            model.parameters(), flags.gradients_norm)
        for p in model.parameters():
            p.data.add_(-flags.lr, p.grad.data)

        #optimizer.step()

        if iteration % 100 == 0:
            print('Epoch: {}/{}'.format(e, flags.epochs),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss_value))

        if iteration % 1000 == 0:
            torch.save(model.state_dict(),
                       os.path.join(flags.checkpoint_path,
                                    'checkpoint_pt/model-{}-{}.pth'.format(flags.output_name, iteration)))
    return model, total_loss, iteration


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(flags.seed)
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)
    model = m.RNNModel(flags.rnn_type, n_vocab, flags.embedding_size, flags.nhid, flags.nlayers)
    if flags.load_checkpoint:
        with open(args.checkpoint, 'rb') as f:
            model = torch.load(f).to(device)
            model.eval()
    print("RNN TYPE: ", flags.rnn_type)
    model = model.to(device)

    if flags.do_train:
        criterion, optimizer = get_loss_and_train_op(model, 0.01)

        total_loss = 0
        iteration = 0
        try:
            for e in range(flags.epochs):
                batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
                model, total_loss, iteration = train(batches, model,device, optimizer, criterion, n_vocab, total_loss, iteration, e)

            # save model after training
            torch.save(model, os.path.join(flags.checkpoint_path, 'model-{}-{}.pth'.format(flags.output_name, 'finished')))

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    if flags.do_predict:
        predict(device, model,n_vocab, vocab_to_int, int_to_vocab, 5)


if __name__ == '__main__':
    main()