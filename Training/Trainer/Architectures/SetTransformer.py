import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from .BaseModel import BaseModel


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
        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        out = out if getattr(self, 'ln0', None) is None else self.ln0(out)
        out = out + F.relu(self.fc_o(out))
        out = out if getattr(self, 'ln1', None) is None else self.ln1(out)
        return out


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
        self.inducing_pts = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        init.xavier_uniform_(self.inducing_pts)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.inducing_pts.repeat(X.size(0), 1, 1), X)
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


class Model(BaseModel):
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
        super().__init__(options)
        self.num_heads = 4
        self.trk_SetTransformer = SetTransformer(self.n_trk_features, self.num_heads, self.output_size).to(self.device)  # Work in progress
        self.calo_SetTransformer = SetTransformer(self.n_calo_features, self.num_heads, self.output_size).to(self.device)  # Work in progress

    def prep_for_forward(self, track_info, track_length, lepton_info, calo_info, calo_length):
        '''
        Preps data for passing through the net
        Args:
            track_info: variable length information about the track
            lepton_info: fixed length information about the lepton
            calo_info: variable length information about caloclusters
            track_length: unpadded length of tracks
            calo_length: unpadded length of caloclusters
        Returns:
            prepared data
        '''
        track_info = track_info.to(self.device)
        lepton_info = lepton_info.to(self.device)
        calo_info = calo_info.to(self.device)
        batch_size, max_n_tracks, n_track_features = track_info.shape
        batch_size, max_n_calos, n_calo_features = calo_info.shape

        return track_info, track_length, lepton_info, calo_info, calo_length

    def forward(self, input_batch):
        r"""Takes prepared data and passes it through the set transformer
            *
            *
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
        Args:
            * input_batch: processed input from prep_for_forward

        Returns:
            the probability of particle beng prompt or heavy flavor
        """
        track_info, track_length, lepton_info, calo_info, calo_length = input_batch
        import pdb
        pdb.set_trace()

        transformed_trk = self.trk_SetTransformer()
        transformed_calo = self.calo_SetTransformer()
        out = self.output_layer(torch.cat([transformed_trk, transformed_calo], dim=1))

        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)

        return out
