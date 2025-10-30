---
Sample treatment
---

A NN is a numerical machine, and numbers that it grinds need to be controlled, often balanced. In this section, we will cover the most common aspects of sample treatment for a NN. Apart from the first which deals with the splitting of the samples, all other subsections can be considered as data pre-processing.

## Splitting: training, validation, and testing samples

We need 3 samples: one to train the NN, another to validate the training, and another where we run the analysis and get results. The main purpose of the validation sample is to see how well the NN classifies the same type of events as either S or B, this, once exposed to events which were not included in the training; we will cover more this aspect below. Therefore, <b>the training and validation samples should be exactly similar</b>: the fraction of the total number of events that they each represent, the composition of the samples, etc. For example: (1) If the training is performed over 25% of the signal sample, then the validation sample has to include a different still 25% of the signal sample as well; the same for the background sample. (2) If the background sample is made of 75% of Wjets and 25% of ttbar events for training, one needs to keep the same percentages for the background validation sample.

## Event normalization

In the case where the values of the input variables vary by several orders of magnitude and/or are different for the various input variables, the adjusting of the NN parameters might be difficult, typically because the same weights will have to cover the possibly wildly different values. It is therefore better to render these values comparable, while they should naturally retain their discriminating power.

→ One possibility is to decrease the order of magnitude of an input variable $x_i$ while taking into account its mean value $<x_i>$ and standard deviation $\sigma_i$: $x'_i = (x_i - <x_i>) / \sigma_i$.

→ Another possibility is to normalize the input variables to the $[-1,+1]$ interval. This has the property of being simple, zero-centered (which can be interesting in some cases), and will be illustrated in among the snippets below.

## Shuffling, seeding


## Event weighting & balancing


## Code snippets

