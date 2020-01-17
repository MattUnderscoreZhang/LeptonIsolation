# -*- coding: utf-8 -*-
"""Driver function for running the neural network

Attributes:
    *--disable-cuda : runs the code only on cpu even if gpu is available
    *--continue-training : loads in a previous model to continue training
"""

from Trainer import trainer as trainer
import argparse
import torch
import time

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="Trainer")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--continue-training", action="store_true", help="Loads in previous model and continues training")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    args.device = torch.device("cpu")


if __name__ == "__main__":

    options = {}
    options["input_data"] = "/public/data/RNN/large_data.root"
    options["run_location"] = "/public/data/RNN/runs"
    options["run_author"] = 'anil_relu_dropout'
    options["tree_name"] = "NormalizedTree"
    options["output_folder"] = "./Outputs/"
    options["model_path"] = options["output_folder"] + "saved_model.pt"
    options["continue_training"] = args.continue_training
    options["architecture_type"] = "GRU"  # RNN, LSTM, GRU, DeepSets
    options["dropout"] = 0.5
    options["track_ordering"] = None  # None, "high-to-low-pt", "low-to-high-pt", "near-to-far", "far-to-near"
    # options["additional_appended_features"] = ["baseline_topoetcone20", "baseline_topoetcone30", "baseline_topoetcone40", "baseline_eflowcone20", "baseline_ptcone20", "baseline_ptcone30", "baseline_ptcone40", "baseline_ptvarcone20", "baseline_ptvarcone30", "baseline_ptvarcone40"]
    options["additional_appended_features"] = []
    options["learning_rate"] = 0.001
    options["training_split"] = 0.7
    options["batch_size"] = 256
    options["n_epochs"] = 30
    options["n_layers"] = 3
    options["hidden_neurons"] = 128
    options["output_neurons"] = 2
    options["device"] = args.device

    t0 = time.time()
    print("number of epochs planned:", options["n_epochs"])
    print("input data:", options["input_data"].split('/')[-1])
    print("batch_size:", options["batch_size"])
    print("device:", args.device)
    print("architecture:", options["architecture_type"])
    trainer.train(options)
    print("total runtime :", time.time() - t0)
    torch.cuda.empty_cache()
