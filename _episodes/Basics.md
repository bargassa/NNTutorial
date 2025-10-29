---
The basics of a NN
---
In this section, we will cover the basic concepts of a Neural Network (NN) classifier. First, we will cover the quantity which is minimized and is guiding the training of a NN, quantity which can also be used by other classifiers. Then we will cover the specifics of a NN classifier, namely its architecture and how one builds its output discriminator as function of the input variables. We will then cover how a NN is trained, and go over the activation functions of a NN. Finally, we will provide code snippets illustrating the mentioned functionalities.

## Loss functions: binary & multi-class

When <b>classifying</b> events, we need to optimize a quantity which quantifies our classification; this quantity can be a loss function. If we are dealing with a binary classification of Signal (S) versus Background (B), we need a measure of how much we have classified signal events as S, and background events as B. The binary cross-entropy, which is one such quantity, is one of the loss functions used for binary studies, and is averaged over N events:

$$\
\begin{align}
L =  - \frac{1}{N} \times \Sigma_{i=1}^{N} [ z(i) \times ln(y(i)) + (1 - z(i)) \times ln(1 - y(i)) ] & (1) &,
\end{align}
\$$

where:
* $z(i)$ is the true classification: it is 1 for S, and 0 for B; this can be viewed as the <b>tag</b> of the event, ie. the prior knowledge that we provide to the classifier for this latter to know which event is S or B.
* $y(i)$ is the classifier's output, with its value between 0 and 1, ie. its <b>prediction</b> for whether an event is S(1) or B(0).

$L$ is reflective of the morphology of the classifier: the measure of the classification, itself a function of the separation achieved by the classifier. In equation (1), the first and second terms are "signal" and "background" term respectively. Indeed:

* $$\ L = \frac{1}{N} \times \Sigma_{i=1}^{N} ln(y(i)). \$$

## Architecture & weights

## Learning/Training of a NN

### Gradient descent

### Adam

### Regularization

## Activation functions

## Number of epochs, batch size

## Code snippets
