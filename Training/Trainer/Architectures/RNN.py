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
        self.h_0 = nn.Parameter(
            torch.zeros(
                self.n_layers, self.batch_size, self.hidden_size
            ).to(self.device)
        )

        self.is_lstm = False
        if options["RNN_type"] == "RNN":
            self.trk_rnn = nn.RNN(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        elif options["RNN_type"] == "LSTM":
            self.is_lstm = True
            self.trk_rnn = nn.LSTM(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        else:
            self.trk_rnn = nn.GRU(
                input_size=self.n_trk_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)

        if options["RNN_type"] == "RNN":
            self.cal_rnn = nn.RNN(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        elif options["RNN_type"] == "LSTM":
            self.is_lstm = True
            self.cal_rnn = nn.LSTM(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)
        else:
            self.cal_rnn = nn.GRU(
                input_size=self.n_cal_features,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=self.n_layers,
                dropout=self.rnn_dropout,
                bidirectional=False,
            ).to(self.device)

        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(self.device)
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.fc_final = nn.Linear(self.hidden_size + self.n_lep_features, self.output_size).to(self.device)
        # self.fc_pooled_lep = nn.Linear(self.hidden_size * 3 + self.n_lep_features, self.output_size).to(self.device)
        # self.fc_lep_info = nn.Linear(self.output_size + self.n_lep_features, self.output_size).to(self.device)
        # self.fc_final = nn.Linear(self.output_size + self.n_lep_features, self.output_size).to(self.device)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, track_info, track_length, lepton_info, cal_info, cal_length):
        r"""Takes data about the event and passes it through:
            * the rnn after padding
            * pool the rnn output to utilize more information than the final layer
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
        self.trk_rnn.flatten_parameters()
        self.cal_rnn.flatten_parameters()

        # moving tensors to adequate device
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)

        # sort and pack padded sequences for tracks
        sorted_n_tracks, sorted_indices_tracks = torch.sort(track_length, descending=True)
        sorted_tracks = track_info[sorted_indices_tracks].to(self.device)
        sorted_n_tracks = sorted_n_tracks.detach().cpu()

        # sort and pack padded sequences for cal
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

        if self.is_lstm:
            output_track, hidden_track, cellstate_track = self.trk_rnn(padded_track_seq, self.h_0)
            output_cal, hidden_cal, cellstate_cal = self.cal_rnn(padded_cal_seq, self.h_0)
        else:
            output_track, hidden_track = self.trk_rnn(padded_track_seq, self.h_0)
            output_cal, hidden_cal = self.cal_rnn(padded_cal_seq, self.h_0)

        output_track, lengths_track = pad_packed_sequence(output_track, batch_first=False)
        output_cal, lengths_cal = pad_packed_sequence(output_cal, batch_first=False)

        # Pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
        avg_pool_track = F.adaptive_avg_pool1d(output_track.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        max_pool_track = F.adaptive_max_pool1d(output_track.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        out_tracks = self.fc_pooled(torch.cat([hidden_track[-1], avg_pool_track, max_pool_track], dim=1))
        # out_tracks = self.dropout(out_tracks)
        avg_pool_cal = F.adaptive_avg_pool1d(output_cal.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        max_pool_cal = F.adaptive_max_pool1d(output_cal.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        out_cal = self.fc_pooled(torch.cat([hidden_cal[-1], avg_pool_cal, max_pool_cal], dim=1))
        # out_cal = self.dropout(out_cal)
        # combining rnn outputs
        out_rnn = self.fc_trk_cal(torch.cat([out_cal[[sorted_indices_cal.argsort()]], out_tracks[[sorted_indices_tracks.argsort()]]], dim=1))
        # out_rnn = self.dropout(out_rnn)
        outp = self.fc_final(torch.cat([out_rnn, lepton_info], dim=1))
        out = self.softmax(outp)

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
