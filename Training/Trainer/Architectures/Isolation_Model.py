import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class InvLinear(nn.Module):
    r"""Permutation invariant linear layer, as described in the
    paper Deep Sets, by Zaheer et al. (https://arxiv.org/abs/1703.06114)
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """

    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        y = torch.zeros(N, self.out_features).to(device)
        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = X * mask.unsqueeze(2).float()
            y = (Z.sum(dim=1) @ self.beta) / sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(2).float()
            y = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


class Model(nn.Module):
    r"""Model class implementing rnn inheriting structure from pytorch nn module

    Attributes:
        options (dict) : configuration for the nn

    Methods:
        forward: steps through the neural net once
        do_train: takes in data and passes the batches to forward to train
        do_eval: runs the neural net on the data after setting it up for evaluation
        get_model: returns the model and its optimizer
    """

    def __init__(self, options):
        super().__init__()
        self.n_layers = options["n_layers"]
        self.n_trk_features = options["n_trk_features"]
        self.n_cal_features = options["n_cal_features"]
        self.hidden_size = options["hidden_neurons"]
        self.n_lep_features = options["n_lep_features"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.rnn_dropout = options["dropout"]
        self.history_logger = SummaryWriter(options["output_folder"])
        self.device = options["device"]
        self.architecture = options["architecture_type"]
        self.h_0 = nn.Parameter(
            torch.zeros(
                self.n_layers, self.batch_size, self.hidden_size
            ).to(self.device)
        )

        if self.architecture == "RNN":
            self.trk_rnn = nn.RNN(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
            self.cal_rnn = nn.RNN(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        elif self.architecture == "LSTM":
            self.trk_rnn = nn.LSTM(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
            self.cal_rnn = nn.LSTM(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        elif self.architecture == "GRU":
            self.trk_rnn = nn.GRU(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
            self.cal_rnn = nn.GRU(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        elif self.architecture == "DeepSets":
            self.feature_extractor = nn.Sequential(
                nn.Linear(options["n_trk_features"], 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 30),
                nn.ReLU(inplace=True)
            )
            self.set = InvLinear(30, 30, bias=True)
            self.output_layer = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(30, self.hidden_size))
        else:
            print("Unrecognized architecture type!")
            exit()

        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(self.device)
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.fc_final = nn.Linear(self.hidden_size + self.n_lep_features, self.output_size).to(self.device)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, track_info, track_length, lepton_info, cal_info, cal_length):
        r"""Takes event data and passes through different layers depending on the architecture.
            RNN / LSTM / GRU:
            * padding variable-length tracks with zeros
            * pool the rnn output to utilize more information than the final layer
            * concatenate all interesting information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
            DeepSets:
            * dense net to get each track to the right latent-space size
            * summation in latent space
            * concatenate all interesting information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
            cal_info: variable length information about caloclusters
            track_length: unpadded length of tracks
            cal_length: unpadded length of caloclusters
        Returns:
            the probability of particle beng prompt or heavy flavor
        """
        # move tensors to either CPU or GPU
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)
        cal_info = cal_info.to(self.device)

        if self.architecture == "DeepSets":
            batch_size, max_n_tracks, n_track_features = track_info.shape
            track_info = track_info.view(batch_size * max_n_tracks, n_track_features)
            intrinsic_tracks = self.feature_extractor(track_info).view(batch_size, max_n_tracks, -1)
            intrinsic_vectors = self.set(intrinsic_tracks, mask=None)
            out = self.output_layer(intrinsic_vectors)
        elif self.architecture in ["RNN", "GRU", "LSTM"]:
            self.trk_rnn.flatten_parameters()
            self.cal_rnn.flatten_parameters()

            # sort and pack padded sequences for tracks and calo clusters
            sorted_n_tracks, sorted_indices_tracks = torch.sort(track_length, descending=True)
            sorted_tracks = track_info[sorted_indices_tracks].to(self.device)
            sorted_n_tracks = sorted_n_tracks.detach().cpu()

            sorted_n_cal, sorted_indices_cal = torch.sort(cal_length, descending=True)
            sorted_cal = cal_info[sorted_indices_cal].to(self.device)
            sorted_n_cal = sorted_n_cal.detach().cpu()

            torch.set_default_tensor_type(torch.FloatTensor)
            padded_track_seq = pack_padded_sequence(sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True)
            padded_cal_seq = pack_padded_sequence(sorted_cal, sorted_n_cal, batch_first=True, enforce_sorted=True)
            if self.device == torch.device("cuda"):
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            padded_track_seq.to(self.device)
            padded_cal_seq.to(self.device)

            if self.architecture == "LSTM":
                output_track, hidden_track, cellstate_track = self.trk_rnn(padded_track_seq, self.h_0)
                output_cal, hidden_cal, cellstate_cal = self.cal_rnn(padded_cal_seq, self.h_0)
            elif self.architecture in ["RNN", "GRU"]:
                output_track, hidden_track = self.trk_rnn(padded_track_seq, self.h_0)
                output_cal, hidden_cal = self.cal_rnn(padded_cal_seq, self.h_0)

            output_track, lengths_track = pad_packed_sequence(output_track, batch_first=False)
            output_cal, lengths_cal = pad_packed_sequence(output_cal, batch_first=False)

            # Pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
            avg_pool_track = F.adaptive_avg_pool1d(output_track.permute(1, 2, 0), 1).view(-1, self.hidden_size)
            max_pool_track = F.adaptive_max_pool1d(output_track.permute(1, 2, 0), 1).view(-1, self.hidden_size)
            out_tracks = self.fc_pooled(torch.cat([hidden_track[-1], avg_pool_track, max_pool_track], dim=1))
            avg_pool_cal = F.adaptive_avg_pool1d(output_cal.permute(1, 2, 0), 1).view(-1, self.hidden_size)
            max_pool_cal = F.adaptive_max_pool1d(output_cal.permute(1, 2, 0), 1).view(-1, self.hidden_size)
            out_cal = self.fc_pooled(torch.cat([hidden_cal[-1], avg_pool_cal, max_pool_cal], dim=1))
            # combining rnn outputs
            out = self.fc_trk_cal(torch.cat([out_cal[[sorted_indices_cal.argsort()]], out_tracks[[sorted_indices_tracks.argsort()]]], dim=1))

        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.softmax(out)

        return out

    def do_train(self, batches, do_training=True):
        r"""Runs the neural net on batches of data passed into it

        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
                                            contains:
                                                * track_info
                                                * lepton_info
                                                * truth
            do_training (bool, True by default): flags whether the model is to be run in
                                                training or evaluation mode

        Returns: total loss, total accuracy, raw results, and all truths

        Notes:
        """
        if do_training:
            self.train()
        else:
            self.eval()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []

        for i, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
            track_info, track_length, cal_info, cal_length, lepton_info, truth = batch
            output = self.forward(track_info, track_length, lepton_info, cal_info, cal_length)
            truth = truth.to(self.device)
            output = output[:, 0]
            loss = self.loss_function(output, truth.float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss)
            predicted = torch.round(output)

            accuracy = float(
                np.array((predicted.data.cpu().detach() ==
                          truth.data.cpu().detach()).sum().float() / len(truth))
            )
            total_acc += accuracy
            raw_results += output.cpu().detach().tolist()
            all_truth += truth.cpu().detach().tolist()
            if do_training is True:
                self.history_logger.add_scalar(
                    "Accuracy/Train Accuracy (Batch)", accuracy, i)
                self.history_logger.add_scalar(
                    "Loss/Train Loss (Batch)", float(loss), i)
            else:
                self.history_logger.add_scalar(
                    "Accuracy/Test Accuracy (Batch)", accuracy, i)
                self.history_logger.add_scalar(
                    "Loss/Test Loss (Batch)", float(loss), i)

        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc, raw_results, all_truth

    def do_eval(self, batches, do_training=False):
        r"""Convienience function for running do_train in evaluation mode

        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
                                            contains:
                                                * track_info
                                                * lepton_info
                                                * truth
            do_training (bool, False by default): flags whether the model is to be run in
                                                training or evaluation mode

        Returns: total loss, total accuracy, raw results, and all truths

        """
        return self.do_train(batches, do_training=False)

    def get_model(self):
        r""" getter function to help easy storage of the model

        Args:
            None

        Returns: the model and its optimizer

        """
        return self, self.optimizer
