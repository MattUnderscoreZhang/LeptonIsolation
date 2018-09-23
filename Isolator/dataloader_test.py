import Loader.loader as loader
from torch.utils.data import DataLoader, Dataset


from DataStructures.LeptonTrackDataset import LeptonTrackDataset


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class Torchdata(Dataset):
    """docstring for Torchdata"""

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index):
        dataiter = next(self.file)
        for i in range(index - 1):
            dataiter = next(self.file)

        return dataiter[2], dataiter[0]

    def __len__(self):
        return len(self.file)

#################
# Main function #
#################


if __name__ == "__main__":

    # set options
    # from Options.default_options import options
    options = {}
    options['n_hidden_output_neurons'] = 8
    options['n_hidden_middle_neurons'] = 8
    options['learning_rate'] = 0.01
    options['training_split'] = 0.9
    options['batch_size'] = 200
    options['n_batches'] = 50
    options['n_layers'] = 8
    options["n_size"] = [8, 8, 8]
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

    train_load = DataLoader(
        data, batch_size=options['batch_size'], collate_fn=my_collate, shuffle=True)
