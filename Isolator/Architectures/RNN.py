'''This file aims to try pytorch rnn module implementation as a
new neural network architecture'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
import numpy as np
import pdb
import time


def Tensor_length(track):
    """Finds the length of the non zero tensor"""
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class RNN(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super(RNN, self).__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.output_size = options["output_size"]
        self.batch_size = options["batch_size"]
        self.learning_rate = options['learning_rate']
        if options['RNN_type'] is 'vanilla':
            self.rnn = nn.RNN(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"])
            
        elif options['RNN_type'] is 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"])
            
        elif options['RNN_type'] is 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"])
            
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)


    def forward(self, tracks):
        self.rnn.flatten_parameters()
        n_tracks = torch.tensor([Tensor_length(tracks[i])
                                 for i in range(len(tracks))])
        sorted_n, indices = torch.sort(n_tracks, descending=True)
        sorted_tracks = tracks[indices]
        output, hidden = self.rnn(pack_padded_sequence(sorted_tracks,
                                                       lengths=sorted_n, batch_first=True))

        out = F.relu(self.fc(hidden[-1]))

        # out=self.softmax(hidden[-1])
        # pdb.set_trace()
        return out

    def accuracy(self, output, truth):
        # predicted, _ = torch.max(output.data, -1)
        predicted = output.data[:, 0]
        acc = (torch.round(predicted).float() ==
               truth.float()).sum().float() / len(truth)
        # if acc < 0.3:
        # pdb.set_trace()
        return acc

    def do_train(self, events, do_training=True):
        if do_training:
            self.train()
        else:
            self.eval()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []
        
        for i, data in enumerate(events, 1):
            track_info, truth = data
            truth = truth[:, 0]
            output = self.forward(track_info)
            loss = self.loss_function(output, truth)
            
            if do_training is True:
                self.optimizer.zero_grad()
                # print("loss:\t",loss)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.data.item()
            total_acc += self.accuracy(output, truth)
            raw_results.append(output.data.detach().numpy())
            all_truth.append(truth.detach().numpy()[0])
        total_loss /= len(events.dataset)
        total_acc = total_acc / len(events.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        total_acc = torch.tensor(total_acc)
        # pdb.set_trace()
        return total_loss.data.item(), total_acc.data.item(), raw_results, torch.tensor(np.array(all_truth))

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)
