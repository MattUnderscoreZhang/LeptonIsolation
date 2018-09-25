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

    target = torch.stack([item[1] for item in batch])
    return [torch.stack(data), target]


class Torchdata(Dataset):
    """docstring for Torchdata"""

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index, length=False):
        '''gets the data at a given index'''
        dataiter = next(self.file)
        for i in range(index - 1):
            # there is probably a better way to do this
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

    from Options.default_options import options
    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False)

    lwt = list(zip(
        leptons_with_tracks['unnormed_leptons'],
        leptons_with_tracks['unnormed_tracks']))

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

    print(rnn.do_train(train_load))
    # for i, data in enumerate(train_load, 1):
    # 	print(data[1])
    #     # rnn.forward(data[0])
