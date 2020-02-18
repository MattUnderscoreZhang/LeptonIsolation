import torch.nn as nn
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

    def forward(self, batch):
        return self.recurrent_forward(batch)


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

    def forward(self, batch):
        return self.recurrent_forward(batch)


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

    def forward(self, batch):
        return self.recurrent_forward(batch)
