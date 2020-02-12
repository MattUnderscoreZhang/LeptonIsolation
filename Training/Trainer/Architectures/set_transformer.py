import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
import numpy as np


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


class MAB(nn.Module):
    '''
    Multihead Attention block as described in https://arxiv.org/pdf/1810.00825.pdf
    '''
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        o = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        o = o if getattr(self, 'ln0', None) is None else self.ln0(o)
        o = o + F.relu(self.fc_o(o))
        o = o if getattr(self, 'ln1', None) is None else self.ln1(o)
        return o


class SAB(nn.Module):
    '''
    Set Attention Block (permutation equivariant)
    SAB(X) := MAB(X,X)
    Takes a set and performs self-attention between the elements in the set returning a set of equal size
    '''
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        '''
        Arguments:
            dim_in: input dimension
            dim_out: output dimension
            num_heads: number of heads
            ln: toggle layer normalization
        '''
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    '''
    Induced Set Attention Block (permutation equivariant)
    ISAB(X) = MAB(X,H) ∈ R^{n×d}, where H = MAB(I,X) ∈ R^{m×d}
    changes the time complexity from O(n^2) to O(nm)
    '''
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        '''
        Arguments:
            dim_in: input dimension
            dim_out: output dimension
            num_heads: number of heads
            num_inds: number of inducing points
            ln: toggle layer normalization
        '''
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    '''
    Pooling by multihead attention (permutation invariant)
    PMAk(Z) = MAB(S,rFF(Z))
    '''
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        '''
        Arguments:
            dim: input and output dimension
            num_heads: number of heads
            num_seeds: number of seed vectors
            ln: toggle layer normalization
        '''
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    '''
    Set transformer consisting of an encoder and decoder
    '''
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        '''
        Arguments:
            dim_input: input dimension
            dim_output: output dimension
            num_outputs: number of seed vectors for PMA
            num_heads: number of heads (default:4)
            num_inds: number of inducing points (default: 32)
            dim_hidden: dimension of hidden layer (default:128)
            ln: toggle layer normalization
        '''
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


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
        self.architecture = options["architecture_type"]
        self.n_trk_features = options["n_trk_features"]
        self.n_calo_features = options["n_calo_features"]
        self.hidden_size = options["hidden_neurons"]
        self.n_lep_features = options["n_lep_features"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["lr"]
        self.batch_size = options["batch_size"]
        self.rnn_dropout = options["dropout"]
        self.device = options["device"]
        if self.architecture == "DeepSets":
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

        elif self.architecture == "SetTransformer":
            self.trk_SetTransformer = SetTransformer(self.n_trk_features, 4, self.output_size).to(self.device)  # Work in progress
            self.calo_SetTransformer = SetTransformer(self.n_calo_features, 4, self.output_size).to(self.device)  # Work in progress
        else:
            print("Unrecognized architecture type!")
            exit()

        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(self.device)
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.fc_final = nn.Linear(self.hidden_size + self.n_lep_features, self.output_size).to(self.device)
        self.relu_final = nn.ReLU(inplace=True)
        self.dropout_final = nn.Dropout(p=options["dropout"])
        self.dropout = nn.Dropout(p=options["dropout"])
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, track_info, track_length, lepton_info, calo_info, calo_length):
        r"""Takes event data and passes through different layers depending on the architecture.
            DeepSets:
            * dense net to get each track to the right latent-space size
            * summation in latent space
            * concatenate all interesting information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
            SetTransformer:
            *
            *
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
            calo_info: variable length information about caloclusters
            track_length: unpadded length of tracks
            calo_length: unpadded length of caloclusters
        Returns:
            the probability of particle beng prompt or heavy flavor
        """
        # move tensors to either CPU or GPU
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)
        calo_info = calo_info.to(self.device)

        if self.architecture == "DeepSets":
            batch_size, max_n_tracks, n_track_features = track_info.shape
            track_info = track_info.view(batch_size * max_n_tracks, n_track_features)
            intrinsic_tracks = self.trk_feature_extractor(track_info).view(batch_size, max_n_tracks, -1)

            batch_size, max_n_calos, n_calo_features = calo_info.shape
            calo_info = calo_info.view(batch_size * max_n_calos, n_calo_features)
            intrinsic_calos = self.calo_feature_extractor(calo_info).view(batch_size, max_n_calos, -1)

            intrinsic_trk_final = self.inv_layer(intrinsic_tracks, track_length)
            intrinsic_calo_final = self.inv_layer(intrinsic_calos, calo_length)
            out = self.output_layer(torch.cat([intrinsic_trk_final, intrinsic_calo_final], dim=1))

        # ~~~~~~~~~ WORK IN PROGRESS ~~~~~~~~~

        if self.architecture == "SetTransformer":
            import pdb; pdb.set_trace()
            batch_size, max_n_tracks, n_track_features = track_info.shape
            unpadded_tracks = [track_info[i][:track_length[i]] for i in range(len(track_length))]

            batch_size, max_n_calos, n_calo_features = calo_info.shape
            unpadded_calo = [calo_info[i][:calo_length[i]] for i in range(len(calo_length))]

            transformed_trk = self.trk_SetTransformer(intrinsic_tracks, track_length)
            transformed_calo = self.calo_SetTransformer(intrinsic_calos, calo_length)
            out = self.output_layer(torch.cat([transformed_trk, transformed_calo], dim=1))

        # ~~~~~~~~~~~ WORK IN PROGRESS ~~~~~~~~

        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        # out = self.dropout_final(out)
        out = self.softmax(out)

        return out

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
            track_info, track_length, calo_info, calo_length, lepton_info, truth, lepton_pT = batch
            output = self.forward(track_info, track_length, lepton_info, calo_info, calo_length)
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
            lep_pT += lepton_pT.cpu().detach().tolist()

        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc, raw_results, all_truth, lep_pT

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
