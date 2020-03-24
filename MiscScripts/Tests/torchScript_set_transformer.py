import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

"""
Testing script for using torchScript for set transformers
"""


class MAB(torch.jit.ScriptModule):
    """
    Multihead Attention block as described in https://arxiv.org/pdf/1810.00825.pdf
    """

    __constants__ = [
        "num_heads",
        "dim_V",
    ]

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

    @torch.jit.script_method
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        out = out  # if getattr(self, "ln0") is None else self.ln0(out)
        out = out + F.relu(self.fc_o(out))
        out = out  # if getattr(self, "ln1") is None else self.ln1(out)
        return out


class SAB(torch.jit.ScriptModule):
    """
    Set Attention Block (permutation equivariant)
    SAB(X) := MAB(X,X)
    Takes a set and performs self-attention between the elements in the set returning a set of equal size
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        """
        Arguments:
            dim_in: input dimension
            dim_out: output dimension
            num_heads: number of heads
            ln: toggle layer normalization
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    @torch.jit.script_method
    def forward(self, X):
        return self.mab(X, X)


class ISAB(torch.jit.ScriptModule):
    """
    Induced Set Attention Block (permutation equivariant)
    ISAB(X) = MAB(X,H) ∈ R^{n×d}, where H = MAB(I,X) ∈ R^{m×d}
    changes the time complexity from O(n^2) to O(nm)
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        """
        Arguments:
            dim_in: input dimension
            dim_out: output dimension
            num_heads: number of heads
            num_inds: number of inducing points
            ln: toggle layer normalization
        """
        super(ISAB, self).__init__()
        self.inducing_pts = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        init.xavier_uniform_(self.inducing_pts)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    @torch.jit.script_method
    def forward(self, X):
        H = self.mab0(self.inducing_pts.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(torch.jit.ScriptModule):
    """
    Pooling by multihead attention (permutation invariant)
    PMAk(Z) = MAB(S,rFF(Z))
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        """
        Arguments:
            dim: input and output dimension
            num_heads: number of heads
            num_seeds: number of seed vectors
            ln: toggle layer normalization
        """
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    @torch.jit.script_method
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(torch.jit.ScriptModule):
    """
    Set transformer consisting of an encoder and decoder
    """

    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        """
        Arguments:
            dim_input: input dimension
            dim_output: output dimension
            num_outputs: number of seed vectors for PMA
            num_heads: number of heads (default:4)
            num_inds: number of inducing points (default: 32)
            dim_hidden: dimension of hidden layer (default:128)
            ln: toggle layer normalization
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    @torch.jit.script_method
    def forward(self, X):
        return self.dec(self.enc(X))


class Model(torch.jit.ScriptModule):
    """
    Set Transformer model class
    """

    __constants__ = [
        "hidden_size",
        "num_heads",
        "n_trk_features",
        "n_calo_features",
        "n_lep_features",
        "output_size",
    ]

    def __init__(self):
        super(Model, self).__init__()
        # self.device = torch.device("cpu")
        self.num_heads: Final[int] = 1
        self.hidden_size: Final[int] = 12
        self.n_trk_features: Final[int] = 6
        self.n_calo_features: Final[int] = 6
        self.n_lep_features: Final[int] = 10
        self.output_size: Final[int] = 2
        self.trk_SetTransformer = SetTransformer(
            self.n_trk_features, num_outputs=self.num_heads, dim_output=self.hidden_size
        )  # .to(self.device)
        self.calo_SetTransformer = SetTransformer(
            self.n_calo_features, self.num_heads, self.hidden_size
        )  # .to(self.device)
        self.output_layer = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )  # .to(self.device)
        self.fc_final = nn.Linear(
            self.hidden_size + self.n_lep_features, self.output_size
        )  # .to(self.device)
        self.relu_final = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    @torch.jit.ignore
    def mock_prep_for_forward(self):
        """
        Preps dummy data for passing through the net

        Returns:
            prepared mock data
        """
        import dummy_data

        track_info = dummy_data.dummy_batch["track_info"]
        track_length = dummy_data.dummy_batch["track_length"]
        lepton_info = dummy_data.dummy_batch["lepton_info"]
        calo_info = dummy_data.dummy_batch["calo_info"]
        calo_length = dummy_data.dummy_batch["calo_length"]

        track_info = track_info  # .to(self.device)
        lepton_info = lepton_info  # .to(self.device)
        calo_info = calo_info  # .to(self.device)

        return track_info, track_length, lepton_info, calo_info, calo_length

    @torch.jit.script_method
    def forward(self, track_info, track_length, lepton_info, calo_info, calo_length):
        r"""Takes prepared data and passes it through the set transformer
            * Set transformer for track and calorimeter information
            * Combine set transformer output with lepton information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
        Args:
            * input_batch: processed input from prep_for_forward

        Returns:
            the probability of particle beng prompt or heavy flavor
        """
        # track_info, track_length, lepton_info, calo_info, calo_length = input_batch

        transformed_trk = self.trk_SetTransformer(track_info)
        transformed_calo = self.calo_SetTransformer(calo_info)
        out = torch.cat(list((transformed_trk, transformed_calo)), dim=2)
        out = self.output_layer(out)
        out = out[:, 0]
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)

        return out

    def save_to_pytorch(self, output_path):
        torch.jit.save(self, output_path)


if __name__ == "__main__":
    # Testing
    model = Model()
    print(model(*model.mock_prep_for_forward()))
    (
        track_info,
        track_length,
        lepton_info,
        calo_info,
        calo_length,
    ) = model.mock_prep_for_forward()
    script = torch.jit.script(
        model, (track_info, track_length, lepton_info, calo_info, calo_length)
    )

    model.to(torch.device("cuda"))
    model.save_to_pytorch("test_set_transformer_gpu.zip")

    # script.save('set_transformer.zip')

    loaded = torch.jit.load("test_set_transformer_gpu.zip")

    print(loaded)
