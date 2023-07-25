### TENSORFLOW GAN METRICS 

**The Metrics which are used for the evalution of Generative Adverserial Networks**
> 1. Frechet Inception Distance 
> 2. Inception Score.

**FID score for Tensorflow**

FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

**Inception Score**

Keeps calssifier intact and don’t use any intermediate values

    Directly see the output of Inception network

    If a score for an exact class is high P(y∣X)P(y∣X)﻿, it means the image is arguably high fidelity since it is easier to recognize and resembles features that is closer to one class.

    Look across many samples and see that the generator is generating many different classes or not P(y)

#### Setup python Package Locally
**1. Clone this repository**
**2. Run setup file**
> python3 setup.py sdist bdist_wheel


#### Usage
You can use the metrics directly by importing the package

     from tensorflow_gan_metrics import FID
     fid_obj = FID.Frechet_Inception_Distance()
     fid_score = fid_obj.call(y_predicts, y_true)


