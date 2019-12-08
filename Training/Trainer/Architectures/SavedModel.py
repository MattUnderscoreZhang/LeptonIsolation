"""This file aims to try pytorch rnn module implementation as a
new neural network architecture"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def hotfix_pack_padded_sequence(
    sorted_tracks, lengths, batch_first=False, enforce_sorted=True
):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(sorted_tracks.device)
        batch_dim = 0 if batch_first else 1
        sorted_tracks = sorted_tracks.index_select(batch_dim, sorted_indices)

    data, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(
        sorted_tracks, lengths, batch_first
    )
    return PackedSequence(data, batch_sizes)


def Tensor_length(track):
    """Finds the length of the non zero tensor"""
    return int(torch.nonzero(track).shape[0] / track.shape[1])


class SavedModel(nn.Module):
    """RNN module implementing pytorch rnn"""

    def __init__(self, options):
        super().__init__()
        self.n_directions = int(options["bidirectional"]) + 1
        self.n_layers = options["n_layers"]
        self.hidden_size = options["hidden_neurons"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["learning_rate"]
        self.batch_size = options["batch_size"]
        self.history_logger = SummaryWriter(options["output_folder"])
        self.device = options["device"]
        self.model = torch.load(self.options["model_path"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def accuracy(self, predicted, truth):
        return torch.from_numpy(
            np.array((predicted == truth.float()).sum().float() / len(truth))
        )

    def do_train(self, batches, do_training=True):
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

            track_info, lepton_info, truth = batch
            # moving tensors to adequate device
            track_info = track_info.to(self.device)
            lepton_info = lepton_info.to(self.device)
            truth = truth[:, 0].to(self.device)

            # setting up for packing padded sequence
            n_tracks = torch.tensor(
                [Tensor_length(track_info[i]) for i in range(len(track_info))]
            )

            sorted_n, indices = torch.sort(n_tracks, descending=True)
            # reodering information according to sorted indices
            sorted_tracks = track_info[indices].to(self.device)
            sorted_leptons = lepton_info[indices].to(self.device)
            padded_seq = hotfix_pack_padded_sequence(
                sorted_tracks, lengths=sorted_n.cpu(), batch_first=True
            )
            output = self.model.forward(padded_seq, sorted_leptons).to(self.device)
            indices = indices.to(self.device)
            loss = self.loss_function(output[:, 0], truth[indices].float())

            if do_training is True:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss)
            predicted = torch.round(output)[:, 0]
            accuracy = float(
                self.accuracy(
                    predicted.data.cpu().detach(), truth.data.cpu().detach()[indices]
                )
            )
            total_acc += accuracy
            raw_results += output[:, 0].cpu().detach().tolist()
            all_truth += truth[indices].cpu().detach().tolist()
            if do_training is True:
                self.history_logger.add_scalar("Accuracy/Train Accuracy", accuracy, i)
                self.history_logger.add_scalar("Loss/Train Loss", float(loss), i)
            else:
                self.history_logger.add_scalar("Accuracy/Test Accuracy", accuracy, i)
                self.history_logger.add_scalar("Loss/Test Loss", float(loss), i)
            for name, param in self.named_parameters():
                self.history_logger.add_histogram(
                    name, param.clone().cpu().data.numpy(), i
                )

        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc, raw_results, all_truth

    def do_eval(self, batches, do_training=False):
        return self.do_train(batches, do_training=False)

    def get_model(self):
        return self.model, self.optimizer
