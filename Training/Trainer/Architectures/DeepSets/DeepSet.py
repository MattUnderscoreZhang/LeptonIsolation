import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .DeepSetLayers import InvLinear


class Model(nn.Module):
    r"""Model class implementing deepsets inheriting structure from pytorch nn module

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

        self.feature_extractor = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30),
            nn.ReLU(inplace=True)
        )
        self.set = InvLinear(30, 30, bias=True)
        self.fc_final = nn.Linear(self.output_size + self.n_lep_features, self.output_size).to(self.device)
        self.output_layer = nn.Sequential(nn.ReLU(inplace=True),
                                          nn.Linear(30, 2))
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _get_mask(sizes, max_size):
        return (torch.arange(max_size).reshape(1, -1).to(sizes.device) < sizes.reshape(-1, 1))

    def forward(self, track_info, track_length, lepton_info, cal_info, cal_length):
        r"""Takes data about the event and passes it through:
            *
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability

        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
        Returns:
           the probability of particle beng prompt or heavy flavor

        """
        import pdb; pdb.set_trace()
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)
        cal_info = cal_info.to(self.device)

        N, S, C, D, _ = X.shape
        h = self.feature_extractor(X.reshape(N, S, C*D*D))
        h = self.set(h, mask=mask)
        outp = self.output_layer(h)
        outp = outp = self.fc_final(torch.cat([outp, leptons_info], dim=1))
        out = self.softmax(p)
        return y

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
