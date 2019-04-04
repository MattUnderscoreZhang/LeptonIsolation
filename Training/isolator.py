from Trainer import trainer
import argparse
import torch


# GPU Compatibility
parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--continue-training', action='store_true',
                    help='loads in previous model and continues training')
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
    options["input_data"] = "/public/data/RNN/lepton_track_data.pkl"
    options["output_folder"] = "../Outputs/"
    options["model_path"] = options["output_folder"] + "saved_optimizer.pt"
    options["continue_training"] = args.continue_training
    print(args.continue_training)
    options["RNN_type"] = "GRU"
    options['learning_rate'] = 0.0003
    options['training_split'] = 0.7
    options['batch_size'] = 500
    options['n_epochs'] = 75
    options['n_layers'] = 3
    options['hidden_neurons'] = 512
    options['output_neurons'] = 2
    options['bidirectional'] = False
    options['device'] = args.device

    trainer.train(options)
