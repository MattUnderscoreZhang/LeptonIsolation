# -*- coding: utf-8 -*-
"""Driver function for tuningg the neural network

Attributes:
    *--disable-cuda : runs the code only on cpu even if gpu is available
    *--continue-training : loads in a previous model to continue training
"""

import torch
import torch.optim as optim
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from filelock import FileLock
import argparse
from ROOT import TFile
from .Architectures.Isolation_Model import Model
from .DataStructures.ROOT_Dataset import ROOT_Dataset, collate
from .Analyzer import Plotter


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="Tuner")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--continue-training", action="store_true", help="Loads in previous model and continues training")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    args.device = torch.device("cpu")

options = {}
options["input_data"] = "/public/data/RNN/small_data.root"
options["run_location"] = "/public/data/RNN/runs"
options["run_label"] = 'anil_relu_dropout'
options["tree_name"] = "NormalizedTree"
options["output_folder"] = "./Outputs/"
options["model_path"] = options["output_folder"] + "saved_model.pt"
options["continue_training"] = args.continue_training
options["architecture_type"] = "GRU"  # RNN, LSTM, GRU, DeepSets
options["dropout"] = 0.3
options["track_ordering"] = "low-to-high-pt"  # None, "high-to-low-pt", "low-to-high-pt", "near-to-far", "far-to-near"
# options["additional_appended_features"] = ["baseline_topoetcone20", "baseline_topoetcone30", "baseline_topoetcone40", "baseline_eflowcone20", "baseline_ptcone20", "baseline_ptcone30", "baseline_ptcone40", "baseline_ptvarcone20", "baseline_ptvarcone30", "baseline_ptvarcone40"]
options["additional_appended_features"] = []
# options["ignore_features"] = ["baseline_eflowcone20", "baseline_eflowcone20_over_pt", "trk_vtx_x", "trk_vtx_y", "trk_vtx_z", "trk_vtx_type"]
options["ignore_features"] = ["baseline_eflowcone20", "baseline_eflowcone20_over_pt", "trk_vtx_type"]
options["learning_rate"] = 0.001
options["training_split"] = 0.7
options["batch_size"] = 256
options["n_epochs"] = 50
options["n_layers"] = 2
options["hidden_neurons"] = 256
options["intrinsic_dimensions"] = 1024  # only matters for deep sets
options["output_neurons"] = 2
options["device"] = args.device


class HyperTune(Trainable):
    """Hyperparameter tuning for lepton isolation model

    Attributes:
        options (dict): configuration for the nn

    Methods:
    """

    def __init__(self, options):
        """Sets up a new agent, or loads a saved agent if training is being resumed.

        Args:
            options (dict): configuration options
        Returns:
            None
        """
        def _load_data(data_filename):
            """Reads the input data and sets up training and test data loaders.

            Args:
                data_filename: ROOT ntuple containing the relevant data
            Returns:
                train_loader, test_loader
            """
            # load data files

            print("Loading data")
            data_file = TFile(data_filename)
            data_tree = getattr(data_file, self.options["tree_name"])
            n_events = data_tree.GetEntries()
            data_file.Close()  # we want each ROOT_Dataset to open its own file and extract its own tree

            # perform class balancing
            print("Balancing classes")
            event_indices = np.array(range(n_events))
            full_dataset = ROOT_Dataset(data_filename, event_indices, self.options, shuffle_indices=False)
            truth_values = [data[-1].bool().item() for data in full_dataset]
            class_0_indices = list(event_indices[truth_values])
            class_1_indices = list(event_indices[np.invert(truth_values)])
            n_each_class = min(len(class_0_indices), len(class_1_indices))
            random.shuffle(class_0_indices)
            random.shuffle(class_1_indices)
            balanced_event_indices = class_0_indices[:n_each_class] + class_1_indices[:n_each_class]
            n_balanced_events = len(balanced_event_indices)
            del full_dataset

            # split test and train
            print("Splitting and processing test and train events")
            random.shuffle(balanced_event_indices)
            n_training_events = int(self.options["training_split"] * n_balanced_events)
            train_event_indices = balanced_event_indices[:n_training_events]
            test_event_indices = balanced_event_indices[n_training_events:]
            train_set = ROOT_Dataset(data_filename, train_event_indices, self.options)
            test_set = ROOT_Dataset(data_filename, test_event_indices, self.options)

            kwargs = {"num_workers": 1, "pin_memory": True} if self.options["device"] == torch.device("cuda") else {}
            # prepare the data loaders
            print("Prepping data loaders")
            train_loader = DataLoader(
                train_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
                **kwargs
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
                **kwargs
            )
            return train_loader, test_loader

    def _setup(self, options):
        self.device = options["device"]
        with FileLock(self.options["input_data"]):
            self.train_loader, self.test_loader = _load_data(self.options["input_data"])
        self.model = Model.to(self.device)


    def _train_iteration(self):

    def _test(self):

    def _train(self):


    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


class CustomStopper(tune.Stopper):
    """docstring for CustomStopper"""
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 5 if args.smoke_test else 100
        if not self.should_stop and result["mean_accuracy"] > 0.96:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop

if __name__ == '__main__':
    stopper = CustomStopper()