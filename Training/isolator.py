# -*- coding: utf-8 -*-
"""Driver function for running the neural network

Attributes:
    *--disable-cuda : runs the code only on cpu even if gpu is available
    *--continue-training : loads in a previous model to continue training

Todo:
    * implement and test deep sets
"""

import argparse
import torch
import time

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# GPU Compatibility
parser = argparse.ArgumentParser(description="Trainer")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--continue-training",
    action="store_true",
    help="loads in previous model and continues training")
parser.add_argument(
    "--deep_sets",
    action="store_true",
    help="uses deep sets")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    args.device = torch.device("cpu")

if args.deep_sets:
    from Trainer import Deep_Set_Trainer as trainer
else:
    from Trainer import trainer as trainer


if __name__ == "__main__":

    options = {}
    # options["input_data"] = "/public/data/RNN/lepton_track_data.pkl"
    options["input_data"] = "/public/data/RNN/data.root"
    options["tree_name"] = "NormalizedTree"
    options["output_folder"] = "./Outputs/"
    options["model_path"] = options["output_folder"] + "saved_model.pt"
    options["continue_training"] = args.continue_training
    options["RNN_type"] = "GRU"
    options["dropout"] = 0.5
    options["track_ordering"] = None  # None, "high-to-low-pt", "low-to-high-pt", "near-to-far", "far-to-near"
    options["learning_rate"] = 0.001
    options["training_split"] = 0.7
    options["batch_size"] = 256
    options["n_epochs"] = 1
    options["n_layers"] = 2
    options["hidden_neurons"] = 128
    options["output_neurons"] = 2
    options["device"] = args.device
    t0 = time.time()
    print("number of epochs planned:", options["n_epochs"])
    print("input data:", options["input_data"].split('/')[-1])
    print("batch_size:", options["batch_size"])
    print("device:", args.device)
    print("deep_sets:", args.deep_sets)
    trainer.train(options)
    print("total runtime :", time.time() - t0)
    torch.cuda.empty_cache()
