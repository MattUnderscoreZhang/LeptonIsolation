import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

################
# Architecture #
################

class RNN(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.n_hidden_neurons = options['n_hidden_neurons']
        self.learning_rate = options['learning_rate']
        self.hidden_layer_1 = nn.Linear(options['n_track_features'] + self.n_hidden_neurons, self.n_hidden_neurons)
        self.hidden_layer_2 = nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.hidden_out_layer = nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.output_layer = nn.Linear(self.n_hidden_neurons, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, track, hidden):
        track = track.view(1, track.size()[0])
        x = torch.cat((track, hidden), 1)
        x = F.tanh(self.hidden_layer_1(x))
        x = F.tanh(self.hidden_layer_2(x))
        hidden = F.tanh(self.hidden_out_layer(x))
        output = F.relu(self.output_layer(x))
        output = self.softmax(output)
        return output, hidden

    def accuracy(self, output, truth):
        _, top_i = output.data.topk(1)
        category = top_i[0][0]
        return (category == truth.data[0])

    def do_train(self, events, do_training=True):
        if do_training:
            self.train()
        else:
            self.eval()
        self.zero_grad()
        total_loss = 0
        total_acc = 0
        for event in events:
            hidden = torch.zeros(1, self.n_hidden_neurons)
            truth, lep_tracks = event
            lepton, tracks = lep_tracks
            for track in tracks:
                output, hidden = self.forward(track, hidden)
            total_loss += self.loss_function(output, truth)
            total_acc += self.accuracy(output, truth)
        total_loss /= len(events)
        total_acc = total_acc.float() / len(events)
        if do_training:
            total_loss.backward()
            for param in self.parameters():
                param.data.add_(-self.learning_rate, param.grad.data)
        return total_loss.data.item(), total_acc.data.item()

    def do_eval(self, events, do_training=False):
        return self.do_train(events, do_training=False)
