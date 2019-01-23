'''This file aims to try pytorch rnn module implementation as a
new neural network architecture'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import argparse

#GPU Compatibility

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    args.device = torch.device('cpu')



def Tensor_length(track):
    """Finds the length of the non zero tensor"""
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class RNN(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super(RNN, self).__init__()
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
                                                       lengths=sorted_n.cpu(),
                                                       batch_first=True))

        combined_out = torch.cat((sorted_leptons, hidden[-1]), dim=1)
        out = self.fc(combined_out)  # add lepton data to the matrix
        out = self.softmax(out)
        return out, indices  # passing indices for reorganizing truth

    def accuracy(self, predicted, truth):
        acc = torch.from_numpy(np.array((predicted == truth.float()).sum().float() / len(truth)))
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
            track_info=track_info.to(args.device)
            lepton_info=lepton_info.to(args.device)
            truth = truth[:, 0].to(args.device)
            output, indices = self.forward(track_info, lepton_info)
            output=output.to(args.device)
            indices=indices.to(args.device)
            loss = self.loss_function(output[:, 0], truth[indices].float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.clone()
            predicted = torch.round(output)[:, 0]
            total_acc += self.accuracy(predicted.data.detach(),
                                       truth.data.detach()[indices]).clone()
            raw_results += list(output[:, 0].data.detach().numpy())
            all_truth += list(truth.detach()[indices].numpy())
        total_loss = total_loss / len(events.dataset) * self.batch_size
        total_acc = total_acc / len(events.dataset) * self.batch_size
        # total_loss = torch.tensor(total_loss)
        # total_acc = torch.tensor(total_acc)
        return total_loss.data.item(), total_acc.data.item(),\
            raw_results, torch.tensor(np.array(all_truth))

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)
