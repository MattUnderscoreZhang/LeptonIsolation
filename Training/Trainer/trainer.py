# -*- coding: utf-8 -*-
"""module for loading in data and training the model. Provides all the utilities
needed for training the model

Attributes:
	*
Todo:
	* implement commenting practices in rest of the file
	* add a way to load models in and train them
		* just loading it it and running doesn't work becuase the model can't
		acess all the functions we have defined for it.

"""

import pickle as pkl
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("..")  # NOQA
from .Architectures.RNN import Model
from .DataStructures.LeptonTrackDataset import Torchdata, collate
from .Analyzer import plot_ROC


class RNN_Trainer:
	"""Model class implementing rnn inheriting structure from pytorch nn module

	Attributes:
		options (dict) : configuration for the nn
		leptons_with_tracks : data to be passed into the nn
		output_folder : path to where the finished network data should be saved

	Methods:
		prepare : prepares the datasets for the model and sets up the model
		make_batches : makes batches for training and testing datasets
		train : trains and saves the model
		test : tests the model and produces a ROC plot
		train_and_test : convienience function for preparing training and testing the model
		save_model : logs all the details of the training and saves it for future reference

	"""

	def __init__(self, options, leptons_with_tracks, output_folder):
		self.options = options
		self.n_events = len(leptons_with_tracks)
		self.n_training_events = int(
			self.options["training_split"] * self.n_events)
		self.leptons_with_tracks = leptons_with_tracks
		self.options["n_track_features"] = len(
			self.leptons_with_tracks[0][1][0])
		self.history_logger = SummaryWriter()
		self.test_truth = []
		self.test_raw_results = []
		self.epoch0 = 0
		self.continue_training = options["continue_training"]

	def prepare(self):
		"""prepares the data for nn use and initializes the neural network

		Args:
			None
		Returns:
			None

		"""
		# split train and test
		np.random.shuffle(self.leptons_with_tracks)
		self.training_events = self.leptons_with_tracks[: self.n_training_events]
		self.test_events = self.leptons_with_tracks[self.n_training_events:]
		# prepare the generators
		self.train_set = Torchdata(self.training_events)
		self.test_set = Torchdata(self.test_events)
		# set up model
		self.model = Model(self.options)
		if self.continue_training is True:

			checkpoint = torch.load(self.options["model_path"])
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.model.optimizer.load_state_dict(
				checkpoint['optimizer_state_dict'])
			self.epoch0 = checkpoint['epoch']

		print("Model parameters:\n{}".format(self.model.parameters))

	def make_batches(self):
		"""makes batches from the training and testing datasets according to 
		hyperparameters specified in options

		Args:
			None
		Returns:
			training_loader, testing_loader

		"""
		training_loader = DataLoader(
			self.train_set,
			batch_size=self.options["batch_size"],
			collate_fn=collate,
			shuffle=True,
			drop_last=True,
		)
		testing_loader = DataLoader(
			self.test_set,
			batch_size=self.options["batch_size"],
			collate_fn=collate,
			shuffle=True,
			drop_last=True,
		)
		return training_loader, testing_loader

	def train(self, Print=True):
		"""trains the model and logs its characteristics for tensorboard

		Args:
			Print (bool, True by default): Specifies whether to print 
											training characteristics on each step
		Returns:
		   train_loss

		"""
		train_loss = 0
		train_acc = 0
		for epoch_n in range(self.options["n_epochs"]):
			training_batches, testing_batches = self.make_batches()
			train_loss, train_acc, _, train_truth = self.model.do_train(
				training_batches
			)
			test_loss, test_acc, _, test_truth = self.model.do_eval(
				testing_batches)
			self.history_logger.add_scalar(
				"Accuracy/Train Accuracy", train_acc, self.epoch0 + epoch_n
			)
			self.history_logger.add_scalar(
				"Accuracy/Test Accuracy", test_acc, self.epoch0 + epoch_n)
			self.history_logger.add_scalar(
				"Loss/Train Loss", train_loss, self.epoch0 + epoch_n)
			self.history_logger.add_scalar(
				"Loss/Test Loss", test_loss, self.epoch0 + epoch_n)
			for name, param in self.model.named_parameters():
				self.history_logger.add_histogram(
					name, param.clone().cpu().data.numpy(), self.epoch0 + epoch_n
				)

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

		return train_loss

	def test(self, data_filename):
		"""Evaluates the model on testing batches

		Args:
			data_filename (string) : datafile for additional data in the roc plot
		Returns:
			None

		"""
		self.test_set.file.reshuffle()
		_, testing_batches = self.make_batches()
		_, _, self.test_raw_results, self.test_truth = self.model.do_eval(
			testing_batches
		)
		ROC_fig = plot_ROC.plot_ROC(
			data_filename, self.test_raw_results, self.test_truth
		)
		self.history_logger.add_figure("ROC", ROC_fig)

	def train_and_test(self, data_filename, do_print=True):
		"""prepares, trains and tests the network

		Args:
			data_filename (string) : datafile for additional data in the roc plot
			do_print (bool, True by default): Specifies whether to print 
											training characteristics on each step
		Returns:
			training loss

		"""
		self.prepare()
		loss = self.train(do_print)
		self.test(data_filename)
		return loss

	def save_model(self, save_path):
		"""saves the model and closes the tensorboard summary writer

		Args:
			save_path (string) : path where the model and its data is to be saved 
		Returns:
			None

		"""
		self.history_logger.export_scalars_to_json(
			self.options["output_folder"] + "/all_scalars.json"
		)
		self.history_logger.close()


def train(options):
	"""Driver function to load data, train model and save results

	Args:
		options (dict) : configuration for the nn
	Returns:
		 None

	"""
	# load data
	data_filename = options["input_data"]
	leptons_with_tracks = pkl.load(
		open(data_filename, "rb"), encoding="latin1")
	options["lepton_size"] = len(leptons_with_tracks["lepton_labels"])
	options["track_size"] = len(leptons_with_tracks["track_labels"])
	lwt = list(
		zip(leptons_with_tracks["normed_leptons"],
			leptons_with_tracks["normed_tracks"])
	)

	# prepare outputs
	output_folder = options["output_folder"]
	if not pathlib.Path(output_folder).exists():
		pathlib.Path(output_folder).mkdir(parents=True)

	# perform training
	RNN_trainer = RNN_Trainer(options, lwt, output_folder)
	RNN_trainer.train_and_test(data_filename)

	# save results
	RNN_trainer.save_model(output_folder)
