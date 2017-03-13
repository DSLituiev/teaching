# Auto-encoders

## Plan
 + Basic definitions
 + Architecture
 + Learning
 + Applications
 + Lab

## Review of basic definitions (1)
 + Generative vs Discriminative models
 
  - Discriminative: 
        learn conditional probability of Y given X:
        what is probability of _labels_ given a _vector of features_?
  - Generative
        learn joint distribution pairs of _labels_ and _feature vectors_
        there is no assymetry, labels are treated same way as features, 

 + Unsupervised learning vs Supervised learning
  - Supervised: a label/target is provided together with a feature vector
  - Unsupervised: no label, only features

 + Representation learning
 
    (Bengio 2014) learning representations of the data that make it easier to extract useful information when building classifiers or other predictors. A good representation is also one that is useful as input to a supervised predictor.One of the challenges of representation learning that distinguishes it from other machine learning tasks such as classification is the difficulty in establishing a clear objective, or target for training.

## Background: compression requires learning joint distribution of features

 + features: 
  e.g. pixels of an image, ordered letters of a text, ordered nucleotides in DNA, stock price at each time step.
 
 + compression: get a shorter description of data
  - compression works by identifying statistical redundancy

 + types of compression
  - lossless compression
  - lossy compression

 + classical examples of compression algorithms for multivariate continous data:
  - PCA
  - ICA
  - autoenconders: linear autoencoder under mean square loss is equivalent to PCA (Goodfellow book)

## Objectives of training of an auto-encoder

 + compression / dimensionality reduction: perform lossy compression of data, distills the number of features (if we regard latent representation as new features)
 + representation learning: learn interdependencies between features (joint distribution)
 + generative model: find a way to generate samples with given joint distribution
 + semi-supervised / transfer learning: find a way to use unlabeled data to improve performance of supervised models.
 + noise reduction: produce a noiseless reconstruction of a noisy signal
 + source separation



## Auto-encoders are generative, unsupervised models

 + require no labels
 + learn joint distribution of features
    (internal structure of the data)
 + learning joint distribution comes with compression of the data



## General Architecture

 + auto-encoders have two basic parts: encoder and decoder
  . *encoder* compresses a multi-dimensional observed data sample to a low-dimensional hidden representation of a sample
  . *decoder* unpacks / interprets / decompresses the low-dimensional hidden / latent representation into the _reconstruction_ of a sample
 + by learning encoding and decoding, the model learns the internal dependencies between features in the real-world data and thus the way to compress the data.


    high dimensional observation ->
    | encoder | ->
    low dimensional hidden / latent representation ->
    | decoder | ->
    high dimensional reconstruction
    

## Applications

## Review of basic definitions (2)

 + Variational inference
    . task: approximate an unknown *real-world* posterior probability distribution P(hidden|observed) of arbitrary shape using some parametrized *model* distribution Q(hidden|observed)
    . given: observed samples that characterize P(observed), family of Q(h|o) functions
    . general approach: maximize marginal _model_ probability distribution of observations Q(o)
    . challenge: direct optimization of marginal model PDF/PMF is intractable
    . specific approach: minimize _variational lower bound_ on difference between the real and model PDFs/PMFs


## Research
 + DRAW: A Recurrent Neural Network For Image Generation. Gregor et al., 2015
    https://arxiv.org/pdf/1502.04623.pdf
 + 


