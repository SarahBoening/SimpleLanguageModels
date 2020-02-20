import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import model
import tokenizer

input_path = ""
output_path = ""
model = "LSTM"
ntokens= 0
emsize = 0
nhid = 0
nlayers = 0
dropout = 0.0
tied = False

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
input_seq = []
target_seq = []
###############################################################################
# Build the model
###############################################################################

model = model.RNNModel(model, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
model = model.to(device)

# Define hyperparameters
n_epochs = 100
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
