import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
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
        self.n_layers = options["n_layers"]
        self.n_trk_features = options["n_trk_features"]
        self.n_calo_features = options["n_calo_features"]
        self.hidden_size = options["hidden_neurons"]
        self.intrinsic_dimensions = options["intrinsic_dimensions"]
        self.n_lep_features = options["n_lep_features"]
        self.output_size = options["output_neurons"]
        self.learning_rate = options["lr"]
        self.batch_size = options["batch_size"]
        self.rnn_dropout = options["dropout"]
        self.device = options["device"]
        self.architecture = options["architecture_type"]
        self.h_0 = nn.Parameter(
            torch.zeros(
                self.n_layers, self.batch_size, self.hidden_size
            ).to(self.device)
        )

        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(self.device)
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.fc_final = nn.Linear(self.hidden_size + self.n_lep_features, self.output_size).to(self.device)
        self.relu_final = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=options["dropout"])
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
        batch["track_info"] = batch["track_info"].to(self.device)
        batch["lepton_info"] = batch["lepton_info"].to(self.device)
        batch["calo_info"] = batch["calo_info"].to(self.device)

        # track_info = batch.track_info
        # track_length = batch.track_length
        # lepton_info = batch.lepton_info
        # calo_info = batch.calo_info
        # calo_length = batch.calo_length

        # move tensors to either CPU or GPU
        # track_info = track_info.to(self.device)
        # lepton_info = lepton_info.to(self.device)
        # calo_info = calo_info.to(self.device)

        # self.trk_rnn.flatten_parameters()
        # self.cal_rnn.flatten_parameters()

        # # sort and pack padded sequences for tracks and calo clusters
        # sorted_n_tracks, sorted_indices_tracks = torch.sort(track_length, descending=True)
        # sorted_tracks = track_info[sorted_indices_tracks].to(self.device)
        # sorted_n_tracks = sorted_n_tracks.detach().cpu()

        # sorted_n_cal, sorted_indices_cal = torch.sort(calo_length, descending=True)
        # sorted_cal = calo_info[sorted_indices_cal].to(self.device)
        # sorted_n_cal = sorted_n_cal.detach().cpu()

        # import pdb; pdb.set_trace()
        # torch.set_default_tensor_type(torch.FloatTensor)
        # padded_track_seq = pack_padded_sequence(sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True)
        # padded_cal_seq = pack_padded_sequence(sorted_cal, sorted_n_cal, batch_first=True, enforce_sorted=True)
        # if self.device == torch.device("cuda"): torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # padded_track_seq.to(self.device)
        # padded_cal_seq.to(self.device)

        # return (padded_track_seq, padded_cal_seq, sorted_indices_tracks, sorted_indices_cal, lepton_info)

        prepped_batch = batch

        return prepped_batch

    def forward(self, input_batch):
        r"""Takes event data and passes through different layers depending on the architecture.
            * pool the rnn output to utilize more information than the final layer
            * concatenate all interesting information
            * a fully connected layer to get it to the right output size
            * a softmax to get a probability
        Args:
            input_batch: event data
        Returns:
            the probability of particle beng prompt or heavy flavor
        """
        padded_track_seq, padded_cal_seq, sorted_indices_tracks, sorted_indices_cal, lepton_info = input_batch
        print("Unimplemented net - please implement in child")
        exit()

    def recurrent_forward(self, batch):
        track_info = batch["track_info"]
        track_length = batch["track_length"]
        lepton_info = batch["lepton_info"]
        calo_info = batch["calo_info"]
        calo_length = batch["calo_length"]

        untrimmed_output_track, untrimmed_hidden_track = self.trk_rnn(track_info, self.h_0)
        untrimmed_output_calo, untrimmed_hidden_calo = self.cal_rnn(calo_info, self.h_0)

        out_tracks = self.trim_rnn_outputs(untrimmed_output_track, track_length)
        out_cal = self.trim_rnn_outputs(untrimmed_output_calo, calo_length)

        # combining rnn outputs
        out = self.fc_trk_cal(torch.cat([out_cal, out_tracks], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)

        return out

    def concat_pooling(self, output_rnn, hidden_rnn):
        # Concat pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
        output_rnn = output_rnn.permute(0, 2, 1)  # converted to BxHxW, W=#words B=batch_size H=#neurons_hidden_layer
        # hidden_rnn already in form LxBxH, L=#layers
        avg_pool_rnn = F.adaptive_avg_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        max_pool_rnn = F.adaptive_max_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        concat_output = torch.cat([hidden_rnn[-1], avg_pool_rnn, max_pool_rnn], dim=1)
        out_rnns = self.fc_pooled(concat_output)
        return out_rnns

    def trim_rnn_outputs(self, output_rnn, sentence_length):
        output_rnn = output_rnn.permute(0, 2, 1)  # converted to BxHxW, W=#words B=batch_size H=hidden_size
        trimmed_out = output_rnn[range(output_rnn.shape[0]), :, (sentence_length-1).tolist()]
        return trimmed_out

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
            output = self.forward(input_batch)
            truth = batch["truth"].to(self.device)
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
            all_truth += batch["truth"].cpu().detach().tolist()
            lep_pT += batch["lepton_pT"].cpu().detach().tolist()

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
