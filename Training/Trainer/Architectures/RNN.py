# -*- coding: utf-8 -*-
"""This module uses pytorch to implement a recurrent neural network capable of
classifying prompt leptons from heavy flavor ones

Attributes:
    *

Todo:
    * test the new pack_padded_sequence implementation on gpu

"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def Tensor_length(track):
    """Finds the length of the non zero tensor

    Args:
        track (torch.tensor): tensor containing the events padded with zeroes at the end

    Returns:
        Length (int) of the tensor were it not zero-padded

    """
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class Model(nn.Module):
    """Model class implementing rnn inheriting structure from pytorch nn module

    Attributes:
        options (dict) : configuration for the nn

    Methods:
        forward: steps through the neural net once
        accuracy: compares predicted values to true values
        do_train: takes in data and passes the batches to forward to train
        do_eval: runs the neural net on the data after setting it up for evaluation
        get_model: returns the model and its optimizer

    """

    def __init__(self, options):
        super().__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.input_size = options["track_size"]
        self.hidden_size = options["hidden_neurons"]
        self.lepton_size = options["lepton_size"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.history_logger = SummaryWriter(options["output_folder"])
        self.device = options["device"]
        self.h_0 = nn.Parameter(
            torch.zeros(
                self.n_layers * self.n_directions, self.batch_size, self.hidden_size
            ).to(self.device)
        )
        self.cellstate = False  # set to true only if lstm

        if options["RNN_type"] == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                bidirectional=options["bidirectional"],
            ).to(self.device)
        elif options["RNN_type"] == "LSTM":
            self.cellstate = True
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                bidirectional=options["bidirectional"],
            ).to(self.device)
        else:
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                bidirectional=options["bidirectional"],
            ).to(self.device)

        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

    def forward(self, padded_seq):
        """Takes a padded sequence and passes it through:
            * the rnn cell
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability

        Args:
            padded_seq (paddedSequence): a collection for lepton track information

        Returns:
           the probability of particle beng prompt or heavy flavor

        """
        self.rnn.flatten_parameters()
        if self.cellstate:
            output, hidden, cellstate = self.rnn(padded_seq, self.h_0)
        else:
            output, hidden = self.rnn(padded_seq, self.h_0)
        out = self.fc(hidden[-1]).to(self.device)
        out = self.softmax(out).to(self.device)
        return out

    def accuracy(self, predicted, truth):
        """Compares the predicted values to the true values

        Args:
            predicted (torch.tensor): predictions from the neural net

        Returns:
            normalized number of accurate predictions

        """
        return torch.from_numpy(
            np.array((predicted == truth.float()).sum().float() / len(truth))
        )

    def do_train(self, batches, do_training=True):
        """runs the neural net on batches of data passed into it

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
            I don't know how the new pack-pad-sequeces works yet

        """
        if do_training:
            self.rnn.train()
        else:
            self.rnn.eval()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []

        for i, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
            track_info, lepton_info, truth = batch

            # moving tensors to adequate device
            track_info = track_info.to(self.device)
            lepton_info = lepton_info.to(self.device)
            truth = truth[:, 0].to(self.device)

            # setting up for packing padded sequence
            n_tracks = torch.tensor(
                [Tensor_length(track_info[i]) for i in range(len(track_info))]
            )
            padded_seq = pack_padded_sequence(
                track_info, n_tracks, batch_first=True, enforce_sorted=False)

            output = self.forward(padded_seq).to(self.device)
            loss = self.loss_function(output[:, 0], truth.float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss)
            predicted = torch.round(output)[:, 0]
            accuracy = float(
                self.accuracy(
                    predicted.data.cpu().detach(), truth.data.cpu().detach()
                )
            )
            total_acc += accuracy
            raw_results += output[:, 0].cpu().detach().tolist()
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
            for name, param in self.named_parameters():
                self.history_logger.add_histogram(
                    name, param.clone().cpu().data.numpy(), i
                )

        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc, raw_results, all_truth

    def do_eval(self, batches, do_training=False):
        """Convienience function for running do_train in evaluation mode

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
        """ getter function to help easy storage of the model

        Args:
            None

        Returns: the model and its optimizer

        """
        return self, self.optimizer
