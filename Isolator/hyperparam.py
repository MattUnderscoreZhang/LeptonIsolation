import skopt.plots as skplot
import skopt
import numpy as np
import pdb
from isolator import RNN_Trainer
import Loader.loader as loader


from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

options = {}
lwt = []

# set up hyperparameter space
space = [Integer(16, 256, name='hidden_neurons'),
         Integer(8, 256, name='n_layers'),
         Real(0.00001, 1.0, name='learning_rate'),
         ]


def option_maker(N_hidden, NL, LR):
    '''Creates dictionary with required hyperparameters'''
    options = {}
    options["RNN_type"] = 'GRU'
    options['learning_rate'] = LR
    options['training_split'] = 0.7
    options['batch_size'] = 100
    options['n_batches'] = 50
    options['n_layers'] = NL
    options['hidden_neurons'] = N_hidden
    options['output_neurons'] = 2
    options['bidirectional'] = False
    return options


@use_named_args(space)
def minimizer(**params):

    options['learning_rate'] = params['learning_rate']
    options['n_layers'] = params['n_layers']
    options['hidden_neurons'] = params['hidden_neurons']
    trainer = RNN_Trainer(options, lwt)
    loss = trainer.train_and_test(Print=False)

    return loss


if __name__ == "__main__":

    # # prepare data
    in_file = "../Data/output.h5"
    save_file = "../Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False, pseudodata=False)

    lwt = list(zip(leptons_with_tracks['normed_leptons'],
                   leptons_with_tracks['normed_tracks']))

    good_leptons = [i[leptons_with_tracks['lepton_labels'].index(
        'ptcone20')] > 0 for i in leptons_with_tracks['unnormed_leptons']]
    lwt = np.array(lwt)[good_leptons]

    options = option_maker(128, 5, 0.0001)
    # pdb.set_trace()
    options['lepton_size'] = len(leptons_with_tracks['lepton_labels'])
    options['track_size'] = len(leptons_with_tracks['track_labels'])

    reg_gp = gp_minimize(minimizer, space, verbose=True)

    print('best score: {}'.format(reg_gp.fun))
    print('best params:')
    print('hidden_neurons: {}'.format(reg_gp.x[0]))
    print('n_layers: {}'.format(reg_gp.x[1]))
    print('learning_rate: {}'.format(reg_gp.x[2]))
