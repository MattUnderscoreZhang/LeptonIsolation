#Environment Setup
To set up a common environment, run "conda env create -f .rnn.yml".
This will create a new Anaconda environment which you can activate with "conda activate rnn".

# Run Instructions
isolator.py is the steering file. It's in charge of performing training, saving the results, and performing analysis.
Edit isolator.py and run with "python isolator.py".

The model will save every 10 epochs to the outputs folder.
To continue training the saved model pass the flad " --continue " while running isolator
