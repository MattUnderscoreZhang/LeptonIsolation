'''This file aims to try pytorch rnn module implementation as a new neural network architechure'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence


class RNN(nn.Module):
    """docstring for RNN"""

    def __init__(self, options):
        super(RNN, self).__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.size = options["n_size"]
        self.batch_size = options["batch_size"]
        self.learning_rate = options['learning_rate']
        self.rnn = nn.RNN(
            input_size=self.size[0], hidden_size=self.size[1],batch_first=True,
            num_layers=self.n_layers, bidirectional=options["bidirectional"])
        self.fc = nn.Linear(self.size[1], self.size[2])
        self.loss_function = nn.CrossEntropyLoss()

    def _init_hidden(self):
        ''' creates hidden layer of given specification'''
        hidden = torch.zeros(self.n_layers,
                             self.batch_size, self.size[1])
        return hidden

    def forward(self, track):

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(pack_sequence(track))

        return output, hidden

    def accuracy(self, output, truth):
        _, top_i = output.data.topk(1)
        category = top_i[0][0]
        return (category == truth.data[0])

    def do_train(self, loader, do_training=True):
        if do_training:
            self.train()
        else:
            self.eval()
        self.zero_grad()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []
        for event in events:
            truth, lepton, tracks = event
            for track in tracks:
                output, hidden = self.forward(track)
            total_loss += self.loss_function(output, truth)
            total_acc += self.accuracy(output, truth)
            raw_results.append(output.detach().numpy()[0][0])
            all_truth.append(truth.detach().numpy()[0])
        total_loss /= len(events)
        total_acc = total_acc.float() / len(events)
        if do_training:
            total_loss.backward()
            for param in self.parameters():
                param.data.add_(-self.learning_rate, param.grad.data)
        return total_loss.data.item(), total_acc.data.item(), raw_results, all_truth
