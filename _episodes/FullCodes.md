---
Full codes
---

The purpose of this part is to give an example of full and working script, for each type of classification (binary and multi-class). These scripts are meant to be illustrative and are naturally improvable. They are commented quite extensively as to guide explain the different aspects of the code. Each code has two type of outputs:
* Plots which provide information about the training (eg. loss, accuracy versus epoch)
* Files which are usable for further investigation of the NN (eg. performance).

Both scripts are based on python, and use Keras. Both run on csv files where each line represents the entries for an event, and where:
* the N first columns are the N input variables: discriminating features, which can either be continuous or discrete variables. Examples: $p_T$ and spatial distribution of various reconstructed particles, invariant masses, charge and identification and isolation variables of leptons, jet- and b-tag multiplicities, b-tagging discriminant distributions, ...
* followed by a column containing the event weight:
* and then by a column containing the string specifying the physical process of the event; the way that the script expects this string is in the NameProcess.root (eg. ttbar.root) form.

This order is important as the numpy manipulation the various elements of the csv file in the script naturally takes it into account. In a certain measure, and for secondary aspects, the two codes have different functionalities, as to illustrate various possible outcomes.


## Binary

The script for the binary NN uses numpy for the manipulation of vectors. It is meant to turn on csv files which provide N=12 input variables. It is integrating all snippets mentioned in previous sections. Its outcomes are:

* 3 plots: TrnVal_loss.png, TrnVal_accuracy.png, TrnVal_roc.png, respectively showing the evolution of the loss, accuracy, and roc curves as a function of the epoch, these for the training and validation samples.
* A file for weights: best_weights.h5 which has the weights of the NN corresponding to the epoch where the validation loss is the lowest, ie. the best weights one can get from the NN training. This file is useful downstream calculations (eg. for calculating the performance of the NN).
* 2 files val_output.csv and test_output.csv, where the outputs of the NN are saved for the validation and test samples, and which are necessary for downstream calculations.

~~~
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.keras import *
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, AlphaDropout
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.mixed_precision import experimental as mixed_precision

#from tensorflow.python.keras.utils import metrics_utils

import numpy
import time
import pandas
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def assure_path_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_data(path, tag):
    """ Read data from CSV
    Args:
      path (str): path to the data
      tag (int): 1 if signal, 0 if background
    Returns:
      np.c_[data, tag] (np.array): array with data and tag
      label (np.array): array with the name of event process
    """
    full_data = np.loadtxt(path, delimiter=",", dtype="str")
    tag = np.ones(len(full_data))*tag
    label = full_data[:, -1]
    data = np.array(full_data[:,:-1], dtype="float")
    return np.c_[data, tag], label

def mae(y_true, y_pred):
    n = len(y_true)
    error = np.abs(y_true - y_pred)
    return error / n

if __name__ == "__main__":
    import argparse
    import sys

    # Uses the "weight" variable of roottuples: provided as (N(input vars)+1)th variable in csv files
    # Passed on weightTrn & weightVal for appropriate samples
    # Weighting each event by these vectors: effectively weighting samples of various XS & Nevts, thus
    # no need of a single "Background.root"
    
    # Input arguments from command line
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-y', '--year', type=str, required=True, help='Year of data being trained')
    parser.add_argument('-c', '--channel', type=str, required=True, help='Final state of data being trained')
    parser.add_argument('-n', '--name', type=str, required=True, help='model name')

    args = parser.parse_args()

    ### Read input arguments
    year = args.year
    channel=args.channel
    name=args.name
    verbose = 0
    if args.verbose:
        verbose = 1
    ###

    ### NN parameters
    activ = "relu"
    learning_rate = 5.e-3
    decay_rate = 0.
    # Each number (here 50) is the number of nodes (Nnode) in each layer.
    # The number of different Nnode (here 2) is the number of layers.
    NodeLayer = "50 50"
    architecture=NodeLayer.split()
    ini = "he_normal" # Function for initialising the random weights of the layer
    n_epochs = 2000
    batch_size = 5000
    dropout_rate = 0.
    ###
    
    # Model's compilation arguments, training parameters and optimizer
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    print(Adam)
    # Define optimizer which has the learning- & decay-rates
    myOpt = Adam(lr=learning_rate, decay=decay_rate)
    # Protect from rounding errors: Number by which the loss-function is multiplied to protect from rounding errors
    #                               dynamically scales the loss to prevent underflow
#    myOpt = mixed_precision.LossScaleOptimizer(myOpt, 1e5)
    compileArgs['optimizer'] = myOpt

    # Creating the directory where the files will be stored
    dirname = year+"_"+channel+"_"+name
    filepath = "models/{}/".format(dirname)
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    # Printing info and starting time
    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()

    ##### Build the model
    model = Sequential()

    # 1st hidden layer: it has as many nodes as provided by architecture[0]: 12 b/c that much input variables
    model.add(Dense(int(architecture[0]), input_dim=12, activation=activ, kernel_initializer=ini))

    i=1
    while i < len(architecture) :
        model.add(Dense(int(architecture[i]), activation=activ, kernel_initializer=ini))
#        model.add(kernel_regularizer=l2(1e-5))
#        model.add(Dropout(dropout_rate))
        i=i+1
    model.add(Dense(1, activation='sigmoid')) # Output layer: 1 node, with sigmoid

    model.compile(**compileArgs)
    model.summary()
    #####
    
    # Read input data
    print("LOADING DATA")
    # Do the labeling for S & B to 1 & 0 for train, val and test samples
    # For train sample: avoid writing the value of label with "_", 
    train_sig, _ = load_data("data/train_sig.csv", 1)
    train_bkg, _ = load_data("data/train_bkg.csv", 0)
    # validation sample
    val_sig, label_val_sig = load_data("data/val_sig.csv", 1)
    val_bkg, label_val_bkg = load_data("data/val_bkg.csv", 0)
    val_label = np.concatenate((label_val_sig, label_val_bkg))
    val_label = np.array([l[:-5] for l in val_label]) # Keep all file/process name except ".root" to identify processes
    # test sample
    test_sig, label_test_sig = load_data("data/test_sig.csv", 1)
    test_bkg, label_test_bkg = load_data("data/test_bkg.csv", 0)
    test_label = np.concatenate((label_test_sig, label_test_bkg))
    test_label = np.array([l[:-5] for l in test_label]) #  Keep all file/process name except ".root" to identify processes
    # Concatenation of variables and weights only
    # NB: could have been done in the general form (concatenate the total object, as above)
    test_data = np.concatenate((test_sig[:, :-2], test_bkg[:, :-2]))
    test_weight = np.concatenate((test_sig[:, -2], test_bkg[:, -2]))

    # Weight normalisation, event balancing:
    # Render the sum of S events equal to sum of B events
    # The totality of weights should be equal, even though the relative weights
    # among B or S signal are correct by weighting each event by the "weight"
    # variable in the roottuples
#    train_sig[:, -2]*=np.sum(train_bkg[:, -2])/np.sum(train_sig[:, -2])
    # AND
    # each sample should be numerically not too small: normalize by the N(B),
    # both for S & B, in both training & validation sample
    train_bkg[:,-2] *=  train_bkg.shape[0] / np.sum(train_bkg[:,-2])
    train_sig[:,-2] *=  train_bkg.shape[0] / np.sum(train_sig[:,-2])
    # Save unweighted values of validation weights, to be written in val_output.csv
    full_val_not_scaled = np.concatenate((val_sig, val_bkg))
    weightVal_not_scaled = full_val_not_scaled[:, -2]
    val_bkg[:,-2] *=  val_bkg.shape[0] / np.sum(val_bkg[:,-2])
    val_sig[:,-2] *=  val_bkg.shape[0] / np.sum(val_sig[:,-2])

    # Concatenate signal and background for train and validation
    full_train = np.concatenate((train_sig, train_bkg))
    full_val = np.concatenate((val_sig, val_bkg))

    # Variables normalisation to [-1,+1]
    # Looping over number of variables:
    #   shape = Dimension of the full_train (variables * Nevts) array
    #   shape[0] gives lines: the dimension is Nevts
    #   shape[1] gives columns: the dimension here is 12 (for input vars.) + 1 (weight) + 1 (tag=0,1)
    for var in range(full_train.shape[1] - 2):
      top = np.max(full_train[:, var]) # checks all lines by value of (variable in) column var
      bot = np.min(full_train[:, var])
      full_train[:, var] = (2*full_train[:, var] - top - bot)/(top - bot)
      full_val[:, var] = (2*full_val[:, var] - top - bot)/(top - bot)
      test_data[:, var] = (2*test_data[:, var] - top - bot)/(top - bot)

    np.random.shuffle(full_train) # Without this line: NN first exposed to S for many events: b/c fullt_train = np.concatenate((train_sig, train_bkg))
    xTrn = full_train[:, :-2] # Input of NN (kinematic variables)
    yTrn = full_train[:, -1] # Target of NN (1 for signal, 0 for background)
    weightTrn = full_train[:, -2]
    xVal = full_val[:, :-2]
    yVal = full_val[:, -1]
    weightVal = full_val[:, -2]

    # Define criterium for best epoch saving
    # Returns the weights as they are at the minimum, b/c of argument of mode
    checkpoint = callbacks.ModelCheckpoint(
      filepath=filepath+"best_weights.h5",
      verbose=1,
      save_weights_only=True,
      monitor="val_loss",
      mode="min",
      save_best_only=True)

    ### Train model ###
    history = model.fit(xTrn, yTrn, validation_data=(xVal,yVal,weightVal), sample_weight=weightTrn,shuffle=True, callbacks=[checkpoint], **trainParams)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history["accuracy"]
    val_acc = history.history['val_accuracy']

    # Saving accuracy and loss values in a pickle file for later plotting
    pickle.dump(acc, open(filepath+"acc.pickle", "wb"))
    pickle.dump(loss, open(filepath+"loss.pickle", "wb"))
    pickle.dump(val_acc, open(filepath+"val_acc.pickle", "wb"))
    pickle.dump(val_loss, open(filepath+"val_loss.pickle", "wb"))

    # Getting the roc curve
    y_pred_Trn = model.predict(xTrn).ravel()
    fpr_Trn, tpr_Trn, thresholds_Trn = roc_curve(yTrn, y_pred_Trn)
    auc_Trn = auc(fpr_Trn, tpr_Trn)
    y_pred_Val = model.predict(xVal).ravel()
    fpr_Val, tpr_Val, thresholds_Val = roc_curve(yVal, y_pred_Val)
    auc_Val = auc(fpr_Val, tpr_Val)

    # Getting errors
#    mae_Trn = mae(yTrn, y_pred_Trn)
#    mae_Val = mae(yVal, y_pred_Val)

    # Plot loss curves
    plt.figure(1)
    plt.plot(loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
#    plt.yscale("log")
    plt.savefig(filepath+"TrnVal_loss.png")
    # Plot accuracy curves
    plt.figure(2)
    plt.plot(acc, label="Training accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
#    plt.yscale("log")
    plt.savefig(filepath+"TrnVal_accuracy.png")
    # Plot ROC curves
    plt.figure(3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Trn, tpr_Trn, label='Training ROC (area = {:.2f})'.format(auc_Trn))
    plt.plot(fpr_Val, tpr_Val, label='Validation ROC (area = {:.2f})'.format(auc_Val))
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(filepath+"TrnVal_roc.png")
#    plt.figure(4)
##    plt.plot([0, 100], 'k--')
#    plt.plot(mae_Trn, label="Training MAE")
#    plt.plot(mae_Val, label="Validation MAE")
#    plt.legend()
#    plt.xlabel('N(events)')
#    plt.ylabel('MAE')
#    plt.savefig(filepath+"TrnVal_mae.png")

    # Time of the training
    if args.verbose:
        print("Training took ", (time.time()-start)//60, " minutes")

    # Getting predictions
    if args.verbose:
        print("Getting predictions")

    model.load_weights(filepath + "best_weights.h5") # Loading best epoch weights

    # Compute and save prediction for best epoch
    trnPredict = model.predict(xTrn)
    valPredict = model.predict(xVal)
    testPredict = model.predict(test_data)
#    np.savetxt(filepath+"val_output.csv", np.c_[valPredict, weightVal, val_label], delimiter=",", fmt="%s")
    np.savetxt(filepath+"val_output.csv", np.c_[valPredict, weightVal_not_scaled, val_label], delimiter=",", fmt="%s")
    np.savetxt(filepath+"test_output.csv", np.c_[testPredict, test_weight, test_label], delimiter=",", fmt="%s")

    # Outputs
#    plt.savefig(filepath+"loss.png")
    val_file = open(filepath + "val_loss.pickle", "rb")
    val = pickle.load(val_file)
    print("Epoch of the best training: ",np.argmin(val))
~~~

## Multiclass

The script for the multi-class NN is classifying 5 different processes (Signal, Wjets, TTbar, Z2nu, Other), and has thus 5 different classes. It uses panda for the manipulation of vectors, and is performing this manipulation slightly differently than in the binary script, but with similar outcomes. It is meant to turn on csv files which provide N=17 input variables. This script is performing 2 loops, the first being nested in the second:

* It is simply running 10 times, letting all random-based processes (initialization, shuffling, seeding) produce 10 different results, which are close in performance.
* It is scanning the decay-rate from $10^{-5}$ to $10^{-3}$.

It goes without saying that these loops can be taken out, or modified. This script has performance calculation capacity embedded in it, and doesn't need a downstream script to be run. The outcomes of the script are:
* A file 17var_fom.csv necessary for possible downstream performance calculations.
* 10 different subdirectories, corresponding to the 10 runs, with each containing:
   * 2 plots: TrnVal_loss.png, TrnVal_accuracy.png, as in the binary case.
   * A file for weights best_weights.h5, as in the binary case.
   * A subdirectory FOM_figures_runs/ which has the performance plots of all 10 runs in it.

As for the event balancing used in the case of multi-class NN, we do an event weighting similar yet different than in the "Event weighting & balancing" section. Interested users are invited to look at slides 10-13 and 23 of [this presentation](https://indico.cern.ch/event/1503118/contributions/6327106/subcontributions/539906/attachments/3037743/5366723/PBargassa_ML3.pdf); referring to these slides, the balancing scheme used in this script is the "SQRT".
