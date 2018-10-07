'''This file aims to try pytorch rnn module implementation as a
new neural network architecture'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb


def Tensor_length(track):
    """Finds the length of the non zero tensor"""
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class RNN(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super(RNN, self).__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.size = options["n_size"]
        self.batch_size = options["batch_size"]
        self.learning_rate = options['learning_rate']
        self.rnn = nn.GRU(
            input_size=self.size[0], hidden_size=self.size[1],
            batch_first=True, num_layers=self.n_layers,
            bidirectional=options["bidirectional"])
        self.fc = nn.Linear(self.size[1], self.size[2])
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _init_hidden(self):
        ''' creates hidden layer of given specification'''
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             self.batch_size, self.size[1])
        return hidden

    def forward(self, tracks):

        self.rnn.flatten_parameters()

        n_tracks = torch.tensor([Tensor_length(tracks[i])
                                 for i in range(len(tracks))])
        sorted_n, indices = torch.sort(n_tracks, descending=True)
        sorted_tracks = tracks[indices]

        output, hidden = self.rnn(pack_padded_sequence(sorted_tracks,
                                                       lengths=sorted_n, batch_first=True))

        fc_output = self.fc(hidden[-1])

        return fc_output

    def accuracy(self, output, truth):

        pdb.set_trace()
        predicted, _ = torch.max(output.data, -1)
        acc = (torch.round(predicted).float() == truth.float()).sum()
        return acc.float() / len(truth)

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
            self.optimizer.zero_grad()
            track_info, truth = data
            output = self.forward(track_info)
            loss = self.loss_function(output, torch.max(truth, 1)[0])
            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.data.item()
            total_acc += self.accuracy(output, truth)
            raw_results.append(output.data.detach().numpy()[0][0])
            all_truth.append(truth.detach().numpy()[0])
        total_loss /= len(events.dataset)
        total_acc = total_acc.float() / len(events.dataset)
        total_loss = torch.tensor(total_loss)
        total_acc = torch.tensor(total_acc)

        return total_loss.data.item(), total_acc.data.item(), raw_results, all_truth

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)
