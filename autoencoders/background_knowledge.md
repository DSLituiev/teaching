

# Representation learning
 
   "learning representations of the data that make it easier to extract useful information when building classifiers or other predictors" (Bengio 2014).
   
   A good representation is also one that is useful as input to a supervised predictor.
   
   One of the challenges of representation learning that distinguishes it from other machine learning tasks such as classification is the difficulty in establishing a clear objective, or target for training.

<span style="color:red">figure: learned filters from CNNs
</span>

---

## Representation learning

<p align="center">
  <img src=tiger_representation.png alt="representation_learning:tiger" width="50%" height="50%"/>
</p>

---


---
# [Review] Generative vs Discriminative
 
## Discriminative ML models
  
+ learn **conditional** probability of labels ($\mathbf{y}$) given features ($\mathbf{x}$)
  - what is probability of a _label_ **given** a _features_?

   $[x_1, x_2, ... x_d] \rightarrow [ y_1, y_2, ... y_m]$


## Generative ML models

+ learn **joint** probability of pairs of _label_ and _feature vectors_
   - what is probability that labels and features **coocur**?
+ no assymetry, labels are treated the same way as features

   $[x_1, x_2, ... x_d,\, y_1, y_2, ... y_m]$ 


_learning implies maximization of the probability of the observed data given model distribution_

---

# [Review] Supervised vs Unsupervised learning

### Supervised learning
- a label/target is provided together with a feature vector
- usually discriminative models (see, Andrew Ng ...)

### Unsupervised learning
 - no label, only features
 - usually generative (exceptions: see ...)

---

# Compression

## Types of compression
  - lossless compression
  - lossy compression ( <- autoencoders )

## Compression algorithms for multivariate continous data
  - Principal component analysis (PCA)
  - Independent component analysis (ICA)
  - Sparse coding
  - Autoenconders: generalization of all above 
  (see Ian Goodfellow book)

---

# Variational inference
 [see tutorial](https://www.cs.jhu.edu/~jason/tutorials/variational.html)

    log p(obs) = 
       = log ∑_{θ,z} p(θ,z,obs)
       = log ∑_{θ,z} q(θ,z) ( p(θ,z,obs)/q(θ,z) )
       = log E_q ( p(θ,z,obs)/q(θ,z) )

       >= E_q log (p(θ,z,obs)/q(θ,z))   by Jensen's inequality
       = E_q log p(θ,z,obs) - Eq log q(θ,z)   
