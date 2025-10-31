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

## Multiclass
