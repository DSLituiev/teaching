# Auto-encoders

 
<p align="center">
  <img src=img/Ouroboros-Free-PNG-Image.png alt="Ouroboros, source=http://www.pngall.com/ouroboros-png"/>
</p>

<p align="center" fontsize=26>
<font size="6">
Dmytro Lituiev
</font>
</p>

---
# Contents

 + Definitions
 + Architecture
 + Learning
 + Applications
 + Lab

---
# [Review] Generative vs Discriminative models
 
## Discriminative: 
  
+ learn conditional probability of labels (Y) given features (X)
+ what is probability of _labels_ given a _vector of features_?


     [x_1, x_2, ..., x_m] --> y


## Generative

+ learn joint distribution pairs of _labels_ and _feature vectors_
+ no assymetry, labels are treated the same way as features


     [x_1, x_2, ..., x_m, y]


_learning implies maximization of the probability of the observed data given model distribution_

---

# [Review] Supervised vs Unsupervised learning

### Supervised learning
a label/target is provided together with a feature vector

### Unsupervised learning
no label, only features

---

## Classifier (supervised)
<p align="center">
  <img src=img/tiger_supervised.png alt="supervised:tiger" width="30%" height="30%"/>
</p>


## Autoencoder (unsupervised)
<p align="center">
  <img src=img/tiger_autoencoder.png alt="autoencoder:tiger" width="50%" height="50%"/>
</p>


---

# Compression

  - getting a shorter description of data
  - compression works by identifying statistical redundancy
  - i.e by learning joint distribution of features


<span style="color:red">figure: groupping of characters 
th -> 0
sh -> 1
ch -> 2
</span>

---


# Compression

## Types of compression
  - lossless compression
  - lossy compression

## Compression algorithms for multivariate continous data
  - Principal component analysis (PCA)
  - Independent component analysis (ICA)
  - Sparse coding
  - Autoenconders: linear autoencoder under mean square loss is equivalent to PCA (Goodfellow book)

---
# Lossy compression with PCA

 + represent a collection of $d$-dimensional vectors by a collection of $m$-dimensional principal components:
 `loadings * input --> scores`
 $W_{1:m} \, \mathbf{x} \rightarrow \, \mathbf{s}_{1:m}$
 `inv(loadings) * scores --> reconstruction`
 $W_{1:m, 1:m}^T \, \mathbf{s}_{1:m} \rightarrow \mathbf{\tilde{x}}$
 + so that $m<d$, thus lossy compression
 + and reconstruction error is minimal
 
   $\mathrm{MSE} = ||\mathbf{\tilde{x}} - \mathbf{x}||^2$
    
---

# Objectives of training of an auto-encoder

 + compression / dimensionality reduction
 + representation / transfer / semisupervised learning
 + data generation
 + noise reduction
 + source separation

---

# Objectives of training of an auto-encoder

 + Compression / dimensionality reduction: perform lossy compression of data, distills the number of features (if we regard latent representation as new features)
 + Representation learning: learn interdependencies between features (joint distribution) that are useful;
 + Semisupervised  [ transfer ] learning: use unlabeled [ any other ] data to improve performance of supervised models
 + Generative model: find a way to generate samples with given joint distribution
 + Noise reduction: produce a noiseless reconstruction of a noisy signal
 + source separation

---
# Semi-supervised learning

<p align="center">
  <img src=img/Erhan2010.png alt="figure from Erhan et al., 2010" width="100%" height="100%"/>
</p>

Learning weights on unlabeled data improves performance in a smaller labeled dataset (Erhan et al., 2010)

---

# Auto-encoders are generative, unsupervised models

 + require no labels
 + learn joint distribution of features
    (internal structure of the data)
 + learning joint distribution comes with compression of the data
 
---

# Architecture

<p align="center">
  <img src=img/tiger_autoencoder.png alt="autoencoder:tiger" width="80%" height="80%"/>
</p>

---

# Architecture

 + auto-encoders have two parts: encoder and decoder
   - **encoder** compresses a multi-dimensional observed data sample to a low-dimensional hidden representation of a sample
   - **decoder** unpacks / interprets / decompresses the low-dimensional hidden / latent representation into the _reconstruction_ of a sample
   
 + by learning encoding and decoding, the model learns the internal **dependencies between features** in the real-world data and thus the way to **compress** the data.

---

# Architecture

<p align="center">
  <img src=img/a_e_graph.png alt="
high dimensional observation ->
| encoder | ->
low dimensional hidden / latent representation ->
| decoder | ->
high dimensional reconstruction" width="80%" height="80%"/>
</p>

---

# Example
```python
from keras.layers import Input, Dense
from keras.models import Model

input_dim = 784
encoding_dim = 32  
# 784 / 32 floats -> compression of factor 24.5

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, 
		activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim,
		activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta',
		    loss='binary_crossentropy')
# ... prepare data: x_train ... 

autoencoder.fit(x_train, x_train, ...)
```
---

# Applications

 + Pre-training / Semi-supervised learning
 + noise reduction
 + generation of new patterns
   - e.g. [Stitchfix](http://multithreaded.stitchfix.com/blog/2015/09/17/deep-style/) 
 + 
 

---
# Variational Autoencoders

Autoencoders with probabilistic flavor

<p align="center">
  <img src=img/v_a_e_graph.png alt="autoencoder:tiger" width="80%" height="80%"/>
</p>

---

# [Review] Variational inference
- **goal**: find a model distribution `Q(observed)` that approximates the real-world distribution of data `P(observed)` 
- **given**: observed samples drawn from real-world distribution `P(observed)`, family of `Q(h|o)` model functions
- **general approach**: maximize marginal _model_ probability distribution of observations `Q(o)`
- **challenge**: direct optimization of marginal model PDF/PMF is intractable
- **specific approach**: minimize _variational lower bound_ on difference between the real and model PDFs/PMFs
- **specific task**: approximate an unknown *real-world* posterior probability distribution `P(hidden|observed)` of an arbitrary shape using some parametrized *model* distribution `Q(hidden|observed)`

---

# Variational Autoencoder: Optimization

- Encode an original vector into a set of vectors representing moments of PDF of $Q(z|x)$, namely mean and variance of $z$.
- Sample Gaussian noise $\epsilon$ and generate hidden states $z = \mu + \sigma \cdot \epsilon$.
- Decode the hidden states into the moments of $P(x|z)$, i.e. predicted mean and variance of observed pixels
- M


---
# Reading

Ian Goodfellow book

[OpenAI blog](
https://openai.com/blog/generative-models/#vae)

[VAE combined with GANs](http://torch.ch/blog/2015/11/13/gan.html)

---

# Research



 + Why Does Unsupervised Pre-training Help Deep Learning? Erhan et al., 2010. [Journal of Machine Learning Research 11](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf)
 + 
 + Auto-encoding variational Bayes. Kingma & Welling 2013.  arXiv:1312.6114 

 + DRAW: A Recurrent Neural Network For Image Generation. Gregor et al., 2015
    https://arxiv.org/pdf/1502.04623.pdf

 + Attend, Infer, Repeat: Fast Scene Understanding with Generative Models. Ali Eslami et al., 2016

	https://arxiv.org/abs/1603.08575
 + Importance Weighted Autoencoders. Burda et al., 2015. 
 	https://arxiv.org/pdf/1509.00519.pdf
    
 + 
---

# Lab resources

+ Keras
https://blog.keras.io/building-autoencoders-in-keras.html

+ .
http://wiseodd.github.io/techblog/2016/12/03/autoencoders/

+ Convolutional Autoencoders in Tensorflow
https://pgaleone.eu/neural-networks/deep-learning/2016/12/13/convolutional-autoencoders-in-tensorflow/

+ Gumbel Max trick VAE [tf]
http://blog.evjang.com/2016/11/tutorial-categorical-variational.html