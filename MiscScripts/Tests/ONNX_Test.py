import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random

batch_size = 256
hidden_size = 256
input_features = 22
max_n_tracks = 12
n_layers = 3

rnn = nn.GRU(
    input_size=input_features,
    hidden_size=hidden_size,
    batch_first=False,
    num_layers=n_layers,
    dropout=0.2,
    bidirectional=False,
)

# packed version - does not work with JIT
track_info = torch.randn(batch_size, max_n_tracks, input_features)
track_length = torch.rand(batch_size).int()
for i in range(batch_size):
    this_track_length = random.randint(1, max_n_tracks)
    track_length[i] = this_track_length
    track_info[i][this_track_length:] = 0

sorted_n_tracks, sorted_indices_tracks = torch.sort(track_length, descending=True)
sorted_tracks = track_info[sorted_indices_tracks]
padded_track_seq = pack_padded_sequence(sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True)

hidden = nn.Parameter(torch.zeros(n_layers, batch_size, hidden_size))
output_track, hidden_track = rnn(padded_track_seq, hidden)
import pdb; pdb.set_trace()

input_batch = (padded_track_seq, hidden)
# torch.onnx.export(rnn, input_batch, "test.onnx", verbose=True)

# padded version - following https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
track_info = []
for i in range(batch_size):
    this_track_length = random.randint(1, max_n_tracks)
    track_info.append(torch.randn(this_track_length, input_features))
padded_track_seq = pad_sequence(track_info)
track_length = [track.size(0) for track in track_info]

output_track, hidden_track = rnn(padded_track_seq)
import pdb; pdb.set_trace()
torch.onnx.export(rnn, padded_track_seq, "test.onnx", verbose=True)
