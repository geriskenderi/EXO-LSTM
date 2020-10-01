import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set up CUDA

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        # Instantiate variables
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, future):
        # Init hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        
        # Init cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        
        # LSTM cell outputs and current states
        out, (ht, ct) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Take the wanted future steps through the linear layer
        predictions = self.linear(out[:, -future:, :])
        
        return predictions

    
class Trainer:
    # A helper class to train the LSTM
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, train_seq, n_epochs):
        for i in range(n_epochs):
            for in_seq, out_seq in train_seq:
                curr_input = in_seq.view(-1, len(in_seq), in_seq.size(1)).to(device)
                curr_out = out_seq.view(-1, len(out_seq), out_seq.size(1)).to(device)

                self.optimizer.zero_grad()
                y_pred = self.model(x=curr_input, future=len(out_seq))
                curr_loss = self.loss_fn(y_pred, curr_out)
                curr_loss.backward()
                self.optimizer.step()

            print(f'epoch: {i+1:3} loss: {curr_loss.item():10.5f}')

        print(f'Loss after training: {curr_loss.item():10.5f}')