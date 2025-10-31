---
Metrics
---

Various metrics exist in order to quantify the performance, and check the sanity of a NN. Below are the main ones.

## Training & Validation losses versus epoch

The training of the NN by definition minimizes the loss, so this latter should decrease versus the epochs for events used for training. So the calculated loss in the validation sample where the weights, as calculated for a given epoch, should have the effect of decreasing the loss in a sample non-exposed to the training. However, a training is never perfect and the loss will never reach nil. As a result, and after a number of epochs, the losses will reach and remain at a minimal and non-zero value. A typical good and realistic example of how training and validation losses should be is given in the plot below: both training and validation losses decrease and plateau after a while, and the validation loss is superior or equal to the training loss, as we do not expect the same weights to yield a better result in the validation sample than for the training sample itself.

> # Figure 5
> <img src="../fig/TrnVal_loss.png" alt="" style="width: 500px;"/>



## True/False positive/negative

> # Figure 6
> <img src="../fig/TFpn.png" alt="" style="width: 500px;"/>

> # Figure 7
> <img src="../fig/Good-vs-Bad_ROC.png" alt="" style="width: 500px;"/>

> # Figure 8
> <img src="../fig/TrnVal_roc.png" alt="" style="width: 500px;"/>

## Accuracy versus epoch

> # Figure 9
> <img src="../fig/TrnVal_accuracy.png" alt="" style="width: 500px;"/>

## Multi-class NN: confusion matrix

> # Figure 10
> <img src="../fig/Confusion_Matrix.png" alt="" style="width: 500px;"/>

## Over-training

> # Figure 11
> <img src="../fig/Perf_NNoutput.png" alt="" style="width: 500px;"/>

## Assess performance of classification in analysis

## Code snippets
