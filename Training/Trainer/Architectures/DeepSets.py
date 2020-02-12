import torch
import torch.nn as nn
from torch.nn import init
import math
from .BaseModel import BaseModel


class InvLinear(nn.Module):
    r"""Permutation invariant linear layer, as described in the
    paper Deep Sets, by Zaheer et al. (https://arxiv.org/abs/1703.06114)
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """

    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, vectors, vector_length):
        r"""
        Reduces set of input vectors to a single output vector.
        Inputs:
            vectors: tensor with shape (batch_size, max_n_vectors, in_features)
        Outputs:
            out: tensor with shape (batch_size, out_features)
        """
        batch_size, max_n_vectors, _ = vectors.shape
        device = vectors.device
        out = torch.zeros(batch_size, self.out_features).to(device)
        mask = torch.Tensor([[1] * i + [0] * (max_n_vectors - i) for i in vector_length.numpy()]).int()  # which elements in vectors are valid

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = vectors * mask.unsqueeze(2).float()
            out = (Z.sum(dim=1) @ self.beta) / sizes

        elif self.reduction == 'sum':
            Z = vectors * mask.unsqueeze(2).float()
            out = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = vectors.clone()
            Z[~mask] = float('-Inf')
            out = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = vectors.clone()
            Z[~mask] = float('Inf')
            out = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


class Model(BaseModel):
    def __init__(self, options):
        super().__init__(options)
        self.trk_feature_extractor = nn.Sequential(
            nn.Linear(options["n_trk_features"], self.intrinsic_dimensions),
            nn.ReLU(inplace=True),
            nn.Linear(300, 30),
            nn.ReLU(inplace=True)
        )
        self.calo_feature_extractor = nn.Sequential(
            nn.Linear(options["n_calo_features"], self.intrinsic_dimensions),
            nn.ReLU(inplace=True),
            nn.Linear(300, 30),
            nn.ReLU(inplace=True)
        )
        self.inv_layer = InvLinear(30, 30, bias=True)
        self.output_layer = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(60, self.hidden_size))

    def prep_for_forward(self, track_info, track_length, lepton_info, calo_info, calo_length):
        # move tensors to either CPU or GPU
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)
        calo_info = calo_info.to(self.device)

        return (track_info, calo_info, track_length, calo_length, lepton_info)

    def forward(self, input_batch):
        track_info, calo_info, track_length, calo_length, lepton_info = input_batch

        batch_size, max_n_tracks, n_track_features = track_info.shape
        track_info = track_info.view(batch_size * max_n_tracks, n_track_features)
        intrinsic_tracks = self.trk_feature_extractor(track_info).view(batch_size, max_n_tracks, -1)

        batch_size, max_n_calos, n_calo_features = calo_info.shape
        calo_info = calo_info.view(batch_size * max_n_calos, n_calo_features)
        intrinsic_calos = self.calo_feature_extractor(calo_info).view(batch_size, max_n_calos, -1)

        intrinsic_trk_final = self.inv_layer(intrinsic_tracks, track_length)
        intrinsic_calo_final = self.inv_layer(intrinsic_calos, calo_length)
        out = self.output_layer(torch.cat([intrinsic_trk_final, intrinsic_calo_final], dim=1))

        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)

        return out
