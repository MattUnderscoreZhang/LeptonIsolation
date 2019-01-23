'''This file aims to try pytorch rnn module implementation as a
new neural network architecture'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


def Tensor_length(track):
    """Finds the length of the non zero tensor"""
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class Net(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super(Net, self).__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.input_size = options["track_size"]
        self.hidden_size = options["hidden_neurons"]
        self.lepton_size = options["lepton_size"]
        self.output_size = options["output_neurons"]
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

        self.fc = nn.Linear(self.hidden_size +
                            self.lepton_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

    def forward(self, tracks, leptons):
        self.rnn.flatten_parameters()
        # list of event lengths
        n_tracks = torch.tensor([Tensor_length(tracks[i])
                                 for i in range(len(tracks))])
        sorted_n, indices = torch.sort(n_tracks, descending=True)
        sorted_tracks = tracks[indices]
        sorted_leptons = leptons[indices]
        output, hidden = self.rnn(pack_padded_sequence(sorted_tracks,
                                                       lengths=sorted_n,
                                                       batch_first=True))

        combined_out = torch.cat((sorted_leptons, hidden[-1]), dim=1)
        out = self.fc(combined_out)  # add lepton data to the matrix
        out = self.softmax(out)
        return out, indices  # passing indices for reorganizing truth

    def accuracy(self, predicted, truth):
        acc = (predicted == truth.float()).sum().float() / len(truth)
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
            self.optimizer.zero_grad()
            track_info, lepton_info, truth = data
            truth = truth[:, 0]
            output, indices = self.forward(track_info, lepton_info)
            loss = self.loss_function(output[:, 0], truth[indices].float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.data.item()
            predicted = torch.round(output)[:, 0]
            total_acc += self.accuracy(predicted.data.detach(),
                                       truth.data.detach()[indices])
            raw_results += list(output[:, 0].data.detach().numpy())
            all_truth += list(truth.detach()[indices].numpy())
        total_loss = total_loss / len(events.dataset) * self.batch_size
        total_acc = total_acc / len(events.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss.data.item(), total_acc.data.item(),\
            raw_results, torch.tensor(np.array(all_truth))

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)

    def get_net(self):
        return self.rnn
