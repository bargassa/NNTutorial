---
Full codes
---

The purpose of this part is to give an example of full and working script, for each type of classification (binary and multi-class). These scripts are meant to be illustrative and are naturally improvable. They are commented quite extensively as to guide explain the different aspects of the code. Each code has two type of outputs:
* Plots which provide information about the training (eg. loss, accuracy versus epoch)
* Files which are usable for further investigation of the NN (eg. performance).

Both scripts are based on python, and use Keras. Both run on csv files where each line represents the entries for an event, and where:
* the N first columns are the N input variables: discriminating features, which can either be continuous or discrete variables. Examples: pT and spatial distribution of various reconstructed particles, invariant masses, charge and identification and isolation variables of leptons, jet- and b-tag multiplicities, b-tagging discriminant distributions, ...
* followed by a column containing the event weight:
* and then by a column containing the string specifying the physical process of the event; the way that the script expects this string is in the NameProcess.root (eg. ttbar.root) form.

This order is important as the numpy manipulation the various elements of the csv file in the script naturally takes it into account. In a certain measure, and for secondary aspects, the two codes have different functionalities, as to illustrate various possible outcomes.


## Binary

## Multiclass
