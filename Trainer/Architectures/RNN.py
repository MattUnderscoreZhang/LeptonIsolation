'''This file aims to try pytorch rnn module implementation as a
new neural network architecture'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import argparse

# GPU Compatibility

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

        if options['RNN_type'] is 'vanilla':
            self.rnn = nn.RNN(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"]).to(args.device)

        elif options['RNN_type'] is 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"]).to(args.device)

        elif options['RNN_type'] is 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size, hidden_size=self.hidden_size,
                batch_first=True, num_layers=self.n_layers,
                bidirectional=options["bidirectional"]).to(args.device)

        self.fc = nn.Linear(self.hidden_size +
                            self.lepton_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, padded_seq, sorted_leptons):
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(padded_seq)

        combined_out = torch.cat((sorted_leptons, hidden[-1]), dim=1)
        out = self.fc(combined_out)  # add lepton data to the matrix
        out = self.softmax(out)
        return out



    


class Net(nn.Module):
    """Super class for the complete neural net"""

    def __init__(self, options):
        super(Net, self).__init__()
        self.options = options
        self.rnn = RNN(options)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.options['learning_rate'])

        # self.rnn = nn.DataParallel(self.rnn).to(args.device) # https://discuss.pytorch.org/t/multi-layer-rnn-with-dataparallel/4450/8


    def forward(self, padded_seq, sorted_leptons):
        x = self.rnn(padded_seq, sorted_leptons)
        return x

    def accuracy(self, predicted, truth):
        acc = torch.from_numpy(
            np.array((predicted == truth.float()).sum().float() / len(truth)))
        return acc

    def do_train(self, events, do_training=True):
        if do_training:
            self.rnn.train()
        else:
            self.rnn.eval()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []

        for i, data in enumerate(events, 1):
            self.optimizer.zero_grad()

            track_info, lepton_info, truth = data
            # moving tensors to adequate device
            track_info = track_info.to(args.device)
            lepton_info = lepton_info.to(args.device)
            truth = truth[:, 0].to(args.device)



            # setting up for packing padded sequence
            n_tracks = torch.tensor([Tensor_length(track_info[i])
                                     for i in range(len(track_info))])

            sorted_n, indices = torch.sort(n_tracks, descending=True)
            # reodering information according to sorted indices
            sorted_tracks = track_info[indices].to(args.device)
            sorted_leptons = lepton_info[indices].to(args.device)
            # import pdb; pdb.set_trace()
            padded_seq = pack_padded_sequence(
                sorted_tracks, lengths=sorted_n.cpu(), batch_first=True)
            output = self.forward(padded_seq, sorted_leptons)
            output = output.to(args.device)
            indices = indices.to(args.device)
            loss = self.loss_function(output[:, 0], truth[indices].float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss)
            predicted = torch.round(output)[:, 0]
            total_acc += float(self.accuracy(predicted.data.detach(),
                                       truth.data.detach()[indices]))
            raw_results += output[:, 0].detach().tolist()
            all_truth += truth[indices].detach().tolist()
        total_loss = total_loss / len(events.dataset) * self.options['batch_size']
        total_acc = total_acc / len(events.dataset) * self.options['batch_size']

        return total_loss, total_acc,\
            raw_results, all_truth

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)
