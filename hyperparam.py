import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle as pkl
import pathlib
from Trainer.Architectures.RNN import Model, hotfix_pack_padded_sequence, Tensor_length
from Trainer.DataStructures.LeptonTrackDataset import Torchdata, collate
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
    '--learning_rate',
    type=float,
    default=0.0001,
    metavar='learning_rate',
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
    cuda = True
else:
    args.device = torch.device('cpu')
    cuda = False


class TrainRNN(Trainable):
    def _setup(self, config):
        self.args = config.pop("args")
        vars(self.args).update(config)

        torch.manual_seed(args.seed)
        kwargs = {}
        if cuda:
            torch.cuda.manual_seed(args.seed)
            kwargs = {'num_workers': 1, 'pin_memory': True}

        self.n_events = len(self.args.dataset)
        self.n_training_events = int(
            self.args.training_split * self.n_events)
        self.leptons_with_tracks = self.args.dataset

        self.training_events = \
            self.leptons_with_tracks[:self.n_training_events]
        self.test_events = self.leptons_with_tracks[self.n_training_events:]
        # prepare the generators
        self.train_set = Torchdata(self.training_events)
        self.test_set = Torchdata(self.test_events)

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.args.batch_size,
            collate_fn=collate, shuffle=True, drop_last=True, **kwargs)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.args.batch_size,
            collate_fn=collate, shuffle=True, drop_last=True, **kwargs)
        self.model = Model(vars(self.args))
        if cuda:
            self.model.cuda()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)

    def _train_iteration(self):
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            track_info, lepton_info, truth = data
            # moving tensors to adequate device
            track_info = track_info.to(args.device)
            lepton_info = lepton_info.to(args.device)
            truth = truth[:, 0].to(args.device)

            # setting up for packing padded sequence
            n_tracks = torch.tensor([Tensor_length(track_info[i])
                                     for i in range(len(track_info))])

            sorted_n, indices = torch.sort(n_tracks, descending=True)
            # reodering information according to sorted indices
            sorted_tracks = track_info[indices].to(args.device)
            sorted_leptons = lepton_info[indices].to(args.device)
            padded_seq = hotfix_pack_padded_sequence(
                sorted_tracks, lengths=sorted_n.cpu(), batch_first=True)
            output = self.model.forward(
                padded_seq, sorted_leptons).to(args.device)
            indices = indices.to(args.device)
            loss = self.model.loss_function(
                output[:, 0], truth[indices].float())
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                track_info, lepton_info, truth = data
                # moving tensors to adequate device
                track_info = track_info.to(args.device)
                lepton_info = lepton_info.to(args.device)
                truth = truth[:, 0].to(args.device)

                # setting up for packing padded sequence
                n_tracks = torch.tensor([Tensor_length(track_info[i])
                                         for i in range(len(track_info))])

                sorted_n, indices = torch.sort(n_tracks, descending=True)
                # reodering information according to sorted indices
                sorted_tracks = track_info[indices].to(args.device)
                sorted_leptons = lepton_info[indices].to(args.device)
                padded_seq = hotfix_pack_padded_sequence(
                    sorted_tracks, lengths=sorted_n.cpu(), batch_first=True)
                output = self.model(
                    padded_seq, sorted_leptons).to(args.device)
                indices = indices.to(args.device)
                test_loss += self.model.loss_function(
                    output[:, 0], truth[indices].float()).item()
                predicted = torch.round(output)[:, 0]
                test_acc += float(self.model.accuracy(predicted.data.cpu().detach(),
                                                      truth.data.cpu().detach()[indices]))

        test_loss = test_loss / len(self.test_loader.dataset)
        test_acc = test_acc / len(self.test_loader.dataset)
        return {"mean_loss": test_loss, "mean_accuracy": test_acc}

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

    return options, lwt


if __name__ == "__main__":

    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    options = {}

    options["input_data"] = "/public/data/RNN/lepton_track_data.pkl"
    options["output_folder"] = "Outputs/HP_tune/"
    options['output_neurons'] = 2
    options['bidirectional'] = False
    arguments = get_dataset(options)
    vars(args).update(arguments[0])
    vars(args).update({'dataset': arguments[1]})
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    ray.init()
    sched = AsyncHyperBandScheduler(
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
                    "learning_rate": tune.sample_from(
                        lambda spec: np.random.uniform(0.0001, 0.1)),

                }
            }
        },
        verbose=2,
        scheduler=sched)
