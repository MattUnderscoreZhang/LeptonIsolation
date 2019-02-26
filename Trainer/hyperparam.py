import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle as pkl
import pathlib
# from .Architectures.RNN import Model
# from .DataStructures.LeptonTrackDataset import Torchdata, collate

from ray.tune import Trainable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Hyperparameter Tuner')
parser.add_argument(
    '--batch-size',
    type=int,
    default=200,
    metavar='N',
    help='input batch size for training (default: 200)')
parser.add_argument(
    '--RNN_type',
    default='GRU',
    metavar='string',
    help='Type of RNN (default: GRU)')
parser.add_argument(
    '--training_split',
    type=float,
    default=0.7,
    metavar='fraction',
    help='ratio of training to testing (default: 0.7)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--n_batches',
    type=int,
    default=50,
    metavar='N',
    help='number of batches (default: 50)')
parser.add_argument(
    '--n_layers',
    type=int,
    default=5,
    metavar='N',
    help='number of rnn layers (default: 5)')
parser.add_argument(
    '--hidden_neurons',
    type=int,
    default=128,
    metavar='N',
    help='size of hidden neurons (default: 128)')
parser.add_argument(
    '--disable-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--smoke-test', action="store_true", help="Finish quickly for testing")

args = parser.parse_args()
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    cuda=True
else:
    args.device = torch.device('cpu')
    cuda=False

class TrainRNN(Trainable):
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)

        torch.manual_seed(args.seed)
        kwargs = {}
        if cuda:
            torch.cuda.manual_seed(args.seed)
            kwargs = {'num_workers': 1, 'pin_memory': True}            


    def _train_iteration(self):
        return

    def _test(self):
        return

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


def get_dataset(options):

    data_file = options['input_data']
    leptons_with_tracks = pkl.load(open(data_file, 'rb'))
    options['lepton_size'] = len(leptons_with_tracks['lepton_labels'])
    options['track_size'] = len(leptons_with_tracks['track_labels'])
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))
    good_leptons = [i[leptons_with_tracks['lepton_labels'].index(
        'ptcone20')] > 0 for i in leptons_with_tracks['unnormed_leptons']]
    lwt = np.array(lwt)[good_leptons]

    # prepare outputs
    output_folder = options['output_folder']
    if not pathlib.Path(output_folder).exists():
        pathlib.Path(output_folder).mkdir(parents=True)

    return options,lwt,output_folder



if __name__ == "__main__":

    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import tune
    from ray.tune.schedulers import HyperBandScheduler

    options = {}

    options["input_data"] = "/public/data/RNN/lepton_track_data.pkl"
    options["output_folder"] = "Outputs/HP_tune/"
    options['learning_rate'] = 0.0001
    options['training_split'] = 0.7
    options['batch_size'] = 200
    options['n_batches'] = 50
    options['n_layers'] = 5
    options['hidden_neurons'] = 128
    options['output_neurons'] = 2
    options['bidirectional'] = False
    arguments=get_dataset(options)

    ray.init()
    sched = HyperBandScheduler(
        time_attr="training_iteration", reward_attr="neg_mean_loss")
    tune.run_experiments(
        {
            "exp": {
                "stop": {
                    "mean_accuracy": 0.95,
                    "training_iteration": 1 if args.smoke_test else 20,
                },
                "resources_per_trial": {
                    "cpu": 8,
                    "gpu": int(not cuda)
                },
                "run": TrainRNN,
                "num_samples": 1 if args.smoke_test else 20,
                "checkpoint_at_end": True,
                "config": {
                    "args": args,
                    "lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.001, 0.1)),
                    "momentum": tune.sample_from(
                        lambda spec: np.random.uniform(0.1, 0.9)),
                }
            }
        },
        verbose=2,
        scheduler=sched)
