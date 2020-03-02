import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random
import onnxruntime

batch_size = 128
hidden_size = 256
input_features = 22
max_n_tracks = 12
n_layers = 3


class AdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x, kernel_size=(inp_size[2], inp_size[3]))


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.avg_pool2d(input=x, kernel_size=(inp_size[2], inp_size[3]))


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=False,
            num_layers=n_layers,
            dropout=0.2,
            bidirectional=False,
        )

    def forward(self, first_input, second_input):
        first_output = self.rnn(first_input)
        second_output = self.rnn(second_input)
        return first_output + second_output


rnn = RNN()

# padded version - following https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
track_info = []
for i in range(batch_size):
    this_track_length = random.randint(1, max_n_tracks)
    track_info.append(torch.randn(this_track_length, input_features))
padded_track_seq = pad_sequence(track_info)
track_length = [track.size(0) for track in track_info]

second_input = nn.Parameter(torch.zeros(max_n_tracks, batch_size, input_features))

rnn.eval()
rnn(padded_track_seq, second_input)
torch.onnx.export(
    rnn,
    (padded_track_seq, second_input),
    "test.onnx",
    verbose=False,
    export_params=True,
    do_constant_folding=True,
    input_names=["first_input", "second_input"],
    output_names=["output"],
    dynamic_axes={
        "first_input": {0: "batch_size"},
        "second_input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# test ONNX

session = onnxruntime.InferenceSession("test.onnx")

inputs = {
    "first_input": padded_track_seq.detach().numpy(),
    "second_input": second_input.detach().numpy(),
}
outputs = session.run(None, inputs)
# print(outputs)
