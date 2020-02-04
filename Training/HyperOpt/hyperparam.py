import torch
from ray import tune
from ray.tune import Trainable
from torch.utils.data import DataLoader
from filelock import FileLock
from ROOT import TFile
from .Architectures.Isolation_Model import Model
from .DataStructures.ROOT_Dataset import ROOT_Dataset, collate
from .Analyzer import Plotter


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

            # prepare the data loaders
            print("Prepping data loaders")
            train_loader = DataLoader(
                train_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
            )
            return train_loader, test_loader

    def _setup(self, config):
        self.train_loader, self.test_loader = _load_data(self.options["input_data"])

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


stopper = CustomStopper()