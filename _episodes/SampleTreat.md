---
Sample treatment
---

A NN is a numerical machine, and numbers that it grinds need to be controlled, often balanced. In this section, we will cover the most common aspects of sample treatment for a NN. Apart from the first which deals with the splitting of the samples, all other subsections can be considered as data pre-processing.

## Splitting: training, validation, and testing samples

We need 3 samples: one to train the NN, another to validate the training, and another where we run the analysis and get results. The main purpose of the validation sample is to see how well the NN classifies the same type of events as either S or B, this, once exposed to events which were not included in the training; we will cover more this aspect below. Therefore, <b>the training and validation samples should be exactly similar</b>: the fraction of the total number of events that they each represent, the composition of the samples, etc. For example: (1) If the training is performed over 25% of the signal sample, then the validation sample has to include a different still 25% of the signal sample as well; the same for the background sample. (2) If the background sample is made of 75% of Wjets and 25% of ttbar events for training, one needs to keep the same percentages for the background validation sample.

## Event normalization


## Shuffling, seeding


## Event weighting & balancing


## Code snippets

