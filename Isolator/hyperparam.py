import skopt.plots as skplot
import skopt
import numpy as np
import pdb
from isolator import RNN_Trainer
import Loader.loader as loader


def option_maker(RNN_type, LR, TS, BS, NB, NL, N_hidden, N_output, bidirectional):
    '''Creates dictionary with required hyperparameters'''
    options = {}
    options["RNN_type"] = RNN_type
    options['learning_rate'] = LR
    options['training_split'] = TS
    options['batch_size'] = BS
    options['n_batches'] = NB
    options['n_layers'] = NL
    options['hidden_neurons'] = N_hidden
    options['output_neurons'] = N_output
    options['bidirectional'] = bidirectional
    return options

def minimizer(LR, TS, BS, NB, NL, N_hidden):

    pass

if __name__ == "__main__":

    # set options
    # from Options.default_options import options as options

    # # prepare data
    in_file = "../Data/output.h5"
    save_file = "../Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False, pseudodata=False)
    options=option_maker(RNN_type='GRU', LR=0.0001, TS=0.7, BS=100, NB=50, NL=5, N_hidden=128, N_output=2, bidirectional=False)
    options['lepton_size'] = len(leptons_with_tracks['lepton_labels'])
    options['track_size'] = len(leptons_with_tracks['track_labels'])
    plot_save_dir = "../Plots/"
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))

    good_leptons = [i[leptons_with_tracks['lepton_labels'].index(
        'ptcone20')] > 0 for i in leptons_with_tracks['unnormed_leptons']]
    lwt = np.array(lwt)[good_leptons]
    trainer = RNN_Trainer(options, lwt, plot_save_dir)
    loss = trainer.train_and_test(Print=False)
    print(loss)
    # writer.export_scalars_to_json("./all_scalars.json")
    # writer.close()
