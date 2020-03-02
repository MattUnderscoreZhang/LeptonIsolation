import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch._jit_internal import Optional

PackedSequence_ = namedtuple(
    "PackedSequence", ["data", "batch_sizes", "sorted_indices", "unsorted_indices"]
)

# type annotation for PackedSequence_ to make it compatible with TorchScript
PackedSequence_.__annotations__ = {
    "data": torch.Tensor,
    "batch_sizes": torch.Tensor,
    "sorted_indices": Optional[torch.Tensor],
    "unsorted_indices": Optional[torch.Tensor],
}


class BaseModel(torch.jit.ScriptModule):
    r"""Model class implementing rnn inheriting structure from pytorch nn module

    Attributes:
        options (dict) : configuration for the nn

    Methods:
        forward: steps through the neural net once
        do_train: takes in data and passes the batches to forward to train
        do_eval: runs the neural net on the data after setting it up for evaluation
        get_model: returns the model and its optimizer
    """

    __constants__ = [
        "n_trk_features",
        "n_calo_features",
        "hidden_size",
        "intrinsic_dimensions",
        "n_lep_features",
        "output_size",
        "learning_rate",
        "batch_size",
    ]

    def __init__(self, options):
        super().__init__(options)
        self.n_trk_features = options["n_trk_features"]
        self.n_calo_features = options["n_calo_features"]
        self.hidden_size = options["hidden_neurons"]
        self.intrinsic_dimensions = options["intrinsic_dimensions"]
        self.n_lep_features = options["n_lep_features"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["lr"]
        self.batch_size = options["batch_size"]
        self.device = options["device"]
        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(
            self.device
        )
        self.fc_final = nn.Linear(
            self.hidden_size + self.n_lep_features, self.output_size
        ).to(self.device)
        self.relu_final = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=options["dropout"])
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.jit.ignore
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
        Returns: total loss, total accuracy, raw results, all truths, and lepton pT
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
        lep_pT = []

        for i, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
            input_batch = self.prep_for_forward(batch)
            output = self.forward(*input_batch)
            truth = batch["truth"].to(self.device)
            output = output[:, 0]
            loss = self.loss_function(output, truth.float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss)
            predicted = torch.round(output)

            accuracy = float(
                np.array(
                    (predicted.data.cpu().detach() == truth.data.cpu().detach())
                    .sum()
                    .float()
                    / len(truth)
                )
            )
            total_acc += accuracy
            raw_results += output.cpu().detach().tolist()
            all_truth += batch["truth"].cpu().detach().tolist()
            lep_pT += batch["lepton_pT"].cpu().detach().tolist()

        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc, raw_results, all_truth, lep_pT

    @torch.jit.ignore
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
        Returns: total loss, total accuracy, raw results, all truths, and lepton pT
        """
        return self.do_train(batches, do_training=False)

    def get_model(self):
        r""" getter function to help easy storage of the model
        Args:
            None
        Returns: the model and its optimizer
        """
        return self, self.optimizer

    def save_to_pytorch(self, output_path):
        torch.jit.save(self, output_path)
