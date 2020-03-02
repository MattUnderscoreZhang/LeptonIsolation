import torch.nn as nn
from .BaseModel import BaseModel
import torch
from torch._jit_internal import Optional
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

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


class RecurrentModel(BaseModel):
    """docstring for RecurrentModel"""

    __constants__ = [
        "n_layers",
        "rnn_dropout",
    ]

    def __init__(self, options):
        super(RecurrentModel, self).__init__(options)
        self.n_layers = options["n_layers"]
        self.rnn_dropout = options["dropout"]
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(
            self.device
        )
        self.h_0 = nn.Parameter(
            torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(
                self.device
            )
        )

    def _pad(
        self,
        data,
        batch_first: bool,
        batch_sizes,
        pad_value: float,
        sorted_indices: Optional[torch.Tensor],
        unsorted_indices: Optional[torch.Tensor],
    ):
        packed_seq = PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)
        return pad_packed_sequence(packed_seq, batch_first, pad_value)

    @torch.jit.ignore
    def prep_for_forward(self, batch):
        r"""Takes event data and preps it for forwarding through the net.
            * padding variable-length tracks and calo hits with zeros
            * sorting events by number of tracks and calo hits
        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
            calo_info: variable length information about caloclusters
            track_length: unpadded length of tracks
            calo_length: unpadded length of caloclusters
        Returns:
            prepared data
        """
        track_info = batch["track_info"].to(self.device)
        lepton_info = batch["lepton_info"].to(self.device)
        calo_info = batch["calo_info"].to(self.device)
        calo_length = batch["calo_length"].to(self.device)
        track_length = batch["track_length"].to(self.device)

        # sort and pack padded sequences for tracks and calo clusters
        sorted_n_tracks, sorted_indices_tracks = torch.sort(
            track_length, descending=True
        )
        sorted_tracks = track_info[sorted_indices_tracks].to(self.device)
        sorted_n_tracks = sorted_n_tracks.detach().cpu()
        sorted_n_cal, sorted_indices_cal = torch.sort(calo_length, descending=True)
        sorted_cal = calo_info[sorted_indices_cal].to(self.device)
        sorted_n_cal = sorted_n_cal.detach().cpu()

        torch.set_default_tensor_type(torch.FloatTensor)
        padded_track_seq = pack_padded_sequence(
            sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True
        )
        padded_cal_seq = pack_padded_sequence(
            sorted_cal, sorted_n_cal, batch_first=True, enforce_sorted=True
        )
        if self.device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        padded_track_seq.to(self.device)
        padded_cal_seq.to(self.device)
        return padded_track_seq, padded_cal_seq, lepton_info

    @torch.jit.script_method
    def forward(
        self,
        padded_track_seq: PackedSequence_,
        padded_cal_seq: PackedSequence_,
        lepton_info,
    ):

        output_track, hidden_track = self.trk_rnn(padded_track_seq, self.h_0)
        output_cal, hidden_cal = self.cal_rnn(padded_cal_seq, self.h_0)

        output_track, lengths_track = pad_packed_sequence(
            output_track, batch_first=False
        )
        output_cal, lengths_cal = pad_packed_sequence(output_cal, batch_first=False)

        out_cal = self.concat_pooling(output_cal, hidden_cal)
        out_tracks = self.concat_pooling(output_track, hidden_track)

        # combining rnn outputs
        out = self.fc_trk_cal(torch.cat([out_cal, out_tracks], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)
        return out

    def concat_pooling(self, output_rnn, hidden_rnn):
        """
        Pools and contatenates rnn output to suggest permutation invariance
        Concat pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
        Args:
            pad_packed_sequence output and final hidden layer
        Returns: processed output
        """
        output_rnn = output_rnn.permute(
            1, 2, 0
        )  # converted to BxHxW, W=#words B=batch_size H=#neurons_hidden_layer
        # hidden_rnn already in form LxBxH, L=#layers
        avg_pool_rnn = F.adaptive_avg_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        max_pool_rnn = F.adaptive_max_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        concat_output = torch.cat([hidden_rnn[-1], avg_pool_rnn, max_pool_rnn], dim=1)
        out_rnns = self.fc_pooled(concat_output)
        return out_rnns


class RNN_Model(RecurrentModel):
    def __init__(self, options):
        super().__init__(options)
        self.trk_rnn = nn.RNN(
            input_size=self.n_trk_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.cal_rnn = nn.RNN(
            input_size=self.n_calo_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)


class LSTM_Model(RecurrentModel):
    def __init__(self, options):
        super().__init__(options)
        self.trk_rnn = nn.LSTM(
            input_size=self.n_trk_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.cal_rnn = nn.LSTM(
            input_size=self.n_calo_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)


class GRU_Model(RecurrentModel):
    def __init__(self, options):
        super().__init__(options)
        self.trk_rnn = nn.GRU(
            input_size=self.n_trk_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.cal_rnn = nn.GRU(
            input_size=self.n_calo_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
