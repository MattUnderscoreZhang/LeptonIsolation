import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from .BaseModel import BaseModel


class RNN_Model(BaseModel):
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

    def forward(self, input_batch):
        padded_track_seq, padded_cal_seq, sorted_indices_tracks, sorted_indices_cal, lepton_info = input_batch
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
        out = self.fc_trk_cal(torch.cat([out_cal[sorted_indices_cal.argsort()], out_tracks[sorted_indices_tracks.argsort()]], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)
        return out


class LSTM_Model(BaseModel):
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

    def forward(self, input_batch):
        padded_track_seq, padded_cal_seq, sorted_indices_tracks, sorted_indices_cal, lepton_info = input_batch
        output_track, hidden_track, cellstate_track = self.trk_rnn(padded_track_seq, self.h_0)
        output_cal, hidden_cal, cellstate_cal = self.cal_rnn(padded_cal_seq, self.h_0)
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
        out = self.fc_trk_cal(torch.cat([out_cal[sorted_indices_cal.argsort()], out_tracks[sorted_indices_tracks.argsort()]], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)
        return out


class GRU_Model(BaseModel):
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

    def forward(self, input_batch):
        padded_track_seq, padded_cal_seq, sorted_indices_tracks, sorted_indices_cal, lepton_info = input_batch
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
        out = self.fc_trk_cal(torch.cat([out_cal[sorted_indices_cal.argsort()], out_tracks[sorted_indices_tracks.argsort()]], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)
        return out
