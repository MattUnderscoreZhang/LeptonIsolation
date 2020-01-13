import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    r"""Model class implementing rnn inheriting structure from pytorch nn module

    Attributes:
        options (dict) : configuration for the nn

    Methods:
        forward: steps through the neural net once
        do_train: takes in data and passes the batches to forward to train
        do_eval: runs the neural net on the data after setting it up for evaluation
        get_model: returns the model and its optimizer
        _tensor_length (private): returns the length of a tensor
    """

    def __init__(self, options):
        super().__init__()
        self.n_layers = options["n_layers"]
        self.n_trk_features = options["n_trk_features"]
        self.hidden_size = options["hidden_neurons"]
        self.n_lep_features = options["n_lep_features"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.dropout = options["dropout"]
        self.history_logger = SummaryWriter(options["output_folder"])
        self.device = options["device"]
        self.h_0 = nn.Parameter(
            torch.zeros(
                self.n_layers, self.batch_size, self.hidden_size
            ).to(self.device)
        )

        self.is_lstm = False
        if options["RNN_type"] == "RNN":
            self.rnn = nn.RNN(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.dropout,
                bidirectional=False,
            ).to(self.device)
        elif options["RNN_type"] == "LSTM":
            self.is_lstm = True
            self.rnn = nn.LSTM(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.dropout,
                bidirectional=False,
            ).to(self.device)
        else:
            self.rnn = nn.GRU(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.dropout,
                bidirectional=False,
            ).to(self.device)

        self.fc_basic = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.output_size).to(self.device)
        self.fc_pooled_lep = nn.Linear(self.hidden_size * 3 + self.n_lep_features, self.output_size).to(self.device)
        self.fc_lep_info = nn.Linear(self.output_size + self.n_lep_features, self.output_size).to(self.device)
        self.fc_final = nn.Linear(self.output_size + self.n_lep_features, self.output_size).to(self.device)

        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, track_info, lepton_info, track_length):
        r"""Takes data about the event and passes it through:
            * the rnn after padding
            * pool the rnn output to utilize more information than the final layer
            * concatenate all interesting information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability

        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
        Returns:
           the probability of particle beng prompt or heavy flavor

        """
        self.rnn.flatten_parameters()

        # moving tensors to adequate device
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)

        # sort and pack padded sequences
        sorted_n_tracks, sorted_indices = torch.sort(track_length, descending=True)

        sorted_tracks = track_info[sorted_indices].to(self.device)
        sorted_leptons = lepton_info[sorted_indices].to(self.device)
        sorted_n_tracks = sorted_n_tracks.detach().cpu()

        torch.set_default_tensor_type(torch.FloatTensor)
        padded_seq = pack_padded_sequence(sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True)
        if self.device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        padded_seq.to(self.device)

        if self.is_lstm:
            output, hidden, cellstate = self.rnn(padded_seq, self.h_0)
        else:
            output, hidden = self.rnn(padded_seq, self.h_0)

        output, lengths = pad_packed_sequence(output)
        # Pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
        avg_pool = F.adaptive_avg_pool1d(output.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        max_pool = F.adaptive_max_pool1d(output.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        outp = self.fc_pooled(torch.cat([hidden[-1], avg_pool, max_pool], dim=1))
        outp = self.fc_final(torch.cat([outp, sorted_leptons], dim=1))
        out = self.softmax(outp)

        return out, sorted_indices

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
            indices have been removed
            I don't know how the new pack-pad-sequences works yet

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
            track_info, lepton_info, truth, track_length = batch
            output, sorted_indices = self.forward(track_info, lepton_info, track_length)
            truth = truth[sorted_indices].to(self.device)
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
                    "Accuracy/Train Accuracy", accuracy, i)
                self.history_logger.add_scalar(
                    "Loss/Train Loss", float(loss), i)
            else:
                self.history_logger.add_scalar(
                    "Accuracy/Test Accuracy", accuracy, i)
                self.history_logger.add_scalar(
                    "Loss/Test Loss", float(loss), i)

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
