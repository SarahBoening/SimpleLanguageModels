import argparse
import time
import math
import os
import random
import re
import torch
import torch.nn as nn
import torch.onnx
import model
import tokenizer


def get_samples(args):
    path = args.data_folder
    all_files = os.listdir(path)
    pattern = re.compile("^[0-9]{3}.*.raw$")
    # remove everything that is not a raw unprocessed file:
    all_files = [f for f in all_files if pattern.match(f)]

    raw_files = [os.path.join(path, f) for f in all_files]
    print(raw_files[:10])
    random.shuffle(raw_files)
    return raw_files


def load_examples(files, tokenizer):
    result = []
    for file in files:
        toks = tokenizer._tokenize(file)
        result.append(tokenizer.convert_tokens_to_ids(toks))
    return result


input_path = ""
output_path = ""
model = "LSTM"
ntokens= 0
emsize = 0
nhid = 0
nlayers = 0
dropout = 0.5
tied = False

# https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py

# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
torch.manual_seed(666)

#if torch.cuda.is_available():
    #if not args.cuda:
        #print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###############################################################################
# Load data
###############################################################################
train_files = get_samples(input_path)
input_seq = load_examples(train_files)
target_seq = []


###############################################################################
# Build the model
###############################################################################

model = model.RNNModel(model, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
model = model.to(device)

# Define hyperparameters
n_epochs = 10
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Training Run
input_seq = input_seq.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
