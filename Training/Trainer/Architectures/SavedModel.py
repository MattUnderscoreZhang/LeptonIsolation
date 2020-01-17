"""This file aims to try pytorch rnn module implementation as a
new neural network architecture

Note: Currently out of date due to change in architecture
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SavedModel(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super().__init__()
        self.n_layers = options["n_layers"]
        self.hidden_size = options["hidden_neurons"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.history_logger = SummaryWriter(options["output_folder"])
        self.device = options["device"]
        self.model = torch.load(self.options["model_path"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def do_train(self, batches, do_training=True):
        r"""Runs the neural net on batches of data passed into it

        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
                                            contains:
                                                * track_info
                                                * lepton_info
                                                * cal_info
                                                * track lengths
                                                * cal lengths
                                                * truth
            do_training (bool, True by default): flags whether the model is to be run in
                                                training or evaluation mode

        Returns: total loss, total accuracy, raw results, and all truths

        Notes:
        """
        if do_training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0
        total_acc = 0
        raw_results = []
        all_truth = []

        for i, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
            track_info, track_length, cal_info, cal_length, lepton_info, truth = batch
            output = self.model.forward(track_info, track_length, lepton_info, cal_info, cal_length)
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
            raw_results += output[:, 0].cpu().detach().tolist()
            all_truth += truth.cpu().detach().tolist()
            if do_training is True:
                self.history_logger.add_scalar("Accuracy/Train Accuracy", accuracy, i)
                self.history_logger.add_scalar("Loss/Train Loss", float(loss), i)
            else:
                self.history_logger.add_scalar("Accuracy/Test Accuracy", accuracy, i)
                self.history_logger.add_scalar("Loss/Test Loss", float(loss), i)

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
        return self.model, self.optimizer
