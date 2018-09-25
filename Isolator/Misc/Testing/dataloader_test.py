import Loader.loader as loader
import torch
from Architectures.test_new_rnn import RNN
from torch.utils.data import DataLoader, Dataset
from DataStructures.LeptonTrackDataset import *


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

    data = Torchdata(lwt)

    rnn = RNN(options)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    train_load = DataLoader(
        data, batch_size=options['batch_size'],
        collate_fn=collate, shuffle=True)

    print(rnn.do_train(train_load))

