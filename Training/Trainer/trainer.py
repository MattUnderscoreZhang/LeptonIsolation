import random
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ROOT import TFile
from .Architectures.RNN import Model
from .DataStructures.ROOT_Dataset import ROOT_Dataset, collate
from .Analyzer import plot_ROC


class RNN_Agent:
    """Model class implementing rnn inheriting structure from pytorch nn module

    Attributes:
        options (dict): configuration for the nn

    Methods:
        get_data_batches: get batched training and testing data from the data loaders
        train_and_test: convienience function for preparing training and testing the model
        save_agent: logs all the details of the training and saves it for future reference
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
            truth_values = [bool(truth.cpu().numpy()[0]) for _, _, truth in full_dataset]
            class_0_indices = event_indices[[i for i in truth_values]]
            class_1_indices = event_indices[[not i for i in truth_values]]
            n_each_class = min(len(class_0_indices), len(class_1_indices))
            random.shuffle(class_0_indices)
            random.shuffle(class_1_indices)
            balanced_event_indices = class_0_indices[:n_each_class] + class_1_indices[:n_each_class]
            n_balanced_events = len(balanced_event_indices)
            del full_dataset

            # split test and train
            print("Splitting test and training events")
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

        self.options = options
        self.model = Model(self.options)
        self.train_loader, self.test_loader = _load_data(self.options["input_data"])
        self.history_logger = SummaryWriter()

        # load previous state if training is resuming
        self.epoch0 = 0
        if self.options["continue_training"] is True:
            saved_agent = torch.load(self.options["model_path"])
            self.model.load_state_dict(saved_agent['model_state_dict'])
            self.model.optimizer.load_state_dict(saved_agent['optimizer_state_dict'])
            self.epoch0 = saved_agent['epoch']
        print("Model parameters:\n{}".format(self.model.parameters))

    def train_and_test(self, do_print=True):
        """Trains and tests the model.

        Args:
            do_print (bool, True by default): Specifies whether to print validation characteristics on each step
        Returns:
            None
        """
        def _train(Print=True):
            """Trains the model and saves training history to history logger.

            Args:
                Print (bool, True by default): Specifies whether to print training characteristics on each step
            Returns:
                None
            """
            for epoch_n in range(self.options["n_epochs"]):
                train_loss, train_acc, _, train_truth = self.model.do_train(self.train_loader)
                test_loss, test_acc, _, test_truth = self.model.do_eval(self.test_loader)
                self.history_logger.add_scalar("Accuracy/Train Accuracy", train_acc, self.epoch0 + epoch_n)
                self.history_logger.add_scalar("Accuracy/Test Accuracy", test_acc, self.epoch0 + epoch_n)
                self.history_logger.add_scalar("Loss/Train Loss", train_loss, self.epoch0 + epoch_n)
                self.history_logger.add_scalar("Loss/Test Loss", test_loss, self.epoch0 + epoch_n)
                for name, param in self.model.named_parameters():
                    self.history_logger.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch0 + epoch_n)

                if Print:
                    print(
                        "Epoch: %03d, Train Loss: %0.4f, Train Acc: %0.4f, "
                        "Test Loss: %0.4f, Test Acc: %0.4f"
                        % (self.epoch0 + epoch_n, train_loss, train_acc, test_loss, test_acc)
                    )
                if (self.epoch0 + epoch_n) % 10 == 0:
                    torch.save({
                        'epoch': self.epoch0 + epoch_n,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.model.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                    }, self.options["model_path"])

        def _test():
            """Evaluates the model on testing batches and saves ROC curve to history logger."""
            _, _, test_raw_results, test_truth = self.model.do_eval(self.test_loader)
            ROC_fig = plot_ROC.plot_ROC(self.options["input_data"], test_raw_results, test_truth)
            self.history_logger.add_figure("ROC", ROC_fig)

        _train(do_print)
        _test()

    def save_agent(self):
        """saves the model and closes the TensorBoard SummaryWriter."""
        if not pathlib.Path(self.options["output_folder"]).exists():
            pathlib.Path(self.options["output_folder"]).mkdir(parents=True)
        self.history_logger.export_scalars_to_json(self.options["output_folder"] + "/all_scalars.json")
        self.history_logger.close()


def set_data_parsing_options(options):
    """Describes the contents of the ROOT data and how it should be parsed.

    Args:
        options (dict): global configuration options
    Returns:
        options (dict): modified with ROOT config options
    """
    truth_features = ["pdgID", "ptcone20", "ptcone30", "ptcone40", "ptvarcone20", "ptvarcone30", "ptvarcone40", "topoetcone20", "topoetcone30", "topoetcone40", "eflowcone20", "PLT", "truth_type"]
    lep_features = ["lep_pT", "lep_eta", "lep_theta", "lep_phi", "lep_d0", "lep_d0_over_sigd0", "lep_z0", "lep_dz0"]
    trk_features = ["trk_lep_dR", "trk_pT", "trk_eta", "trk_phi", "trk_d0", "trk_z0", "trk_lep_dEta", "trk_lep_dPhi", "trk_lep_dD0", "trk_lep_dZ0", "trk_charge", "trk_chi2", "trk_nIBLHits", "trk_nPixHits", "trk_nPixHoles", "trk_nPixOutliers", "trk_nSCTHits", "trk_nSCTHoles", "trk_nTRTHits"]

    options["truth_features"] = truth_features
    options["lep_features"] = lep_features
    options["trk_features"] = trk_features
    options["lepton_size"] = len(lep_features)
    options["track_size"] = len(trk_features)

    return options


def train(options):
    """Driver function to load data, train model and save results.

    Args:
        options (dict) : configuration for the nn
    Returns:
         None
    """
    options = set_data_parsing_options(options)
    agent = RNN_Agent(options)
    agent.train_and_test()
    agent.save_agent()
