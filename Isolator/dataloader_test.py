import Loader.loader as loader
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from Architectures.test_new_rnn import RNN
from DataStructures.LeptonTrackDataset import LeptonTrackDataset


def my_collate(batch):
    '''pads the data with 0'''
    length = torch.tensor([len(item[0]) for item in batch])
    max_size = length.max()
    data = [nn.ZeroPad2d((0, 0, 0, max_size - len(item[0])))
            (item[0]) for item in batch]

    target = [item[1] for item in batch]
    return [torch.stack(data), target]


class Torchdata(Dataset):
    """docstring for Torchdata"""

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index, length=False):
        dataiter = next(self.file)
        for i in range(index - 1):
            dataiter = next(self.file)
        if length == False:
            return dataiter[2], dataiter[0]
        else:
            return dataiter[2], len(dataiter[2])

    def __len__(self):
        return len(self.file)

#################
# Main function #
#################


if __name__ == "__main__":

    options = {}
    options['bidirectional'] = False
    options['n_hidden_output_neurons'] = 8
    options['n_hidden_middle_neurons'] = 8
    options['learning_rate'] = 0.01
    options['training_split'] = 0.9
    options['batch_size'] = 200
    options['n_batches'] = 50
    options['n_layers'] = 8
    options["n_size"] = [2, 2, 2]
    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False)

    # print(leptons_with_tracks)

    lwt = list(zip(
        leptons_with_tracks['unnormed_leptons'],
        leptons_with_tracks['unnormed_tracks']))
    # cones.compare_ptcone_and_etcone(lwt, plot_save_dir)

    # perform training
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))

    dataset = LeptonTrackDataset(lwt)

    data = Torchdata(lwt)

    rnn = RNN(options)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_load = DataLoader(
        data, batch_size=options['batch_size'], collate_fn=my_collate, shuffle=True)

    for i, data in enumerate(train_load, 1):
        # print(len(data[0]))
        rnn.forward(data[0])
