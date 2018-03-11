import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

################
# DATA LOADING #
################

def load_data(data_file):

    with open(data_file, 'rb') as f:
        un = pickle._Unpickler(f)
        un.encoding = 'latin1'
        data = un.load()
    lep_feature_dict = data[0][0]
    track_feature_dict = data[0][1]
    data.pop(0)

    return(lep_feature_dict, track_feature_dict, data)

################
# ARCHITECTURE #
################

class RNN(nn.Module):

    def __init__(self, n_track_features, n_hidden_neurons):
        super().__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.hidden_layer = nn.Linear(n_track_features + n_hidden_neurons, n_hidden_neurons)
        self.output_layer = nn.Linear(n_track_features + n_hidden_neurons, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_values, hidden_values):
        input_values = input_values.view(1, input_values.size()[0])
        combined = torch.cat((input_values, hidden_values), 1)
        hidden = self.hidden_layer(combined)
        output = self.output_layer(combined)
        output = self.softmax(output)
        return output, hidden

    def train(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        rnn.zero_grad()
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for param in rnn.parameters():
            param.data.add_(-learning_rate, param.grad.data)
        return output, loss.data[0]

    def evaluate(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        return output, loss.data[0]

###############
# MAIN SCRIPT #
###############

if __name__ == "__main__":

    data_file = '../../Data/LepIso/393407.pkl'
    lep_feature_dict, track_feature_dict, data = load_data(data_file)
    n_track_features = len(track_feature_dict)

    n_hidden_neurons = 128
    learning_rate = 0.005
    training_split = 0.66

    n_events = len(data)
    n_training_events = int(training_split * n_events)
    training_data = data[:n_training_events]
    test_data = data[n_training_events:]

    rnn = RNN(n_track_features, n_hidden_neurons)
    training_loss = 0
    for lep_n in range(len(training_data)):
        truth = training_data[lep_n][lep_feature_dict['lepIso_lep_isolated']]
        if (truth == -1): # unrecognized type
            continue
        tracks = training_data[lep_n][0]
        truth = Variable(torch.LongTensor([truth]))
        output, loss = rnn.train(truth, Variable(torch.FloatTensor(tracks)))
        training_loss += loss
        if lep_n % 100 == 0:
            print('%d%% trained, average loss is %.4f' % (lep_n / len(training_data) * 100, training_loss / (lep_n+1)))

    test_loss = 0
    for lep_n in range(len(test_data)):
        truth = test_data[lep_n][lep_feature_dict['lepIso_lep_isolated']]
        if (truth == -1): # unrecognized type
            continue
        tracks = test_data[lep_n][0]
        truth = Variable(torch.LongTensor([truth]))
        output, loss = rnn.evaluate(truth, Variable(torch.FloatTensor(tracks)))
        test_loss += loss
    print('Test loss is %.4f' % (test_loss / (lep_n+1)))
