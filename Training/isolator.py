from Trainer import trainer
import argparse
import torch


# GPU Compatibility
parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    args.device = torch.device('cpu')


if __name__ == "__main__":
    '''python isolator.py'''
    options = {}

    options["input_data"] = "/public/data/RNN/lepton_track_data_track_pt_not_weighted.pkl"
    options["output_folder"] = "../Outputs/Runs/Test/"

    options["RNN_type"] = "RNN"
    options['learning_rate'] = 0.0005
    options['training_split'] = 0.7
    options['batch_size'] = 500
    options['n_epochs'] = 50
    options['n_layers'] = 3
    options['hidden_neurons'] = 256
    options['output_neurons'] = 2
    options['bidirectional'] = False
    options['device'] = args.device

    trainer.train(options)
