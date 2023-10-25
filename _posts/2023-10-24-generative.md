---
title: 'Mathematical View of Generative Models'
date: 2023-10-24
permalink: /posts/2023/10/generative/
tags:
  - deep learning
  - generative models
  - mathemtics
---

Generative models, powered by AI, is a big trend now for stuff synthesis (texts, images, and even more specialize usages). This post probes into why and how they are working from a mathematical perspective.

- [To start with: Important Theorems \& Corollaries](#to-start-with-important-theorems--corollaries)
  - [Tweedie Estimator](#tweedie-estimator)
  - [Score Matching](#score-matching)
  - [Langevin Dynamics](#langevin-dynamics)
- [References](#references)


# To start with: Important Theorems & Corollaries

Generally, in Bayesian inference, we would like to estimate parameter $\theta$ from observations $x$. For example, if we adopt MSE to measure the error between our estimation and the ground truth, then we have

$$
L=\mathbb{E}[(\hat{\theta}(x)-\theta)^2],
$$

whose optimization equals to solving the expected value of the posterior distribution

$$
\hat{\theta}(x)=\mathbb{E}[\hat{\theta}\mid x]=\int \theta \times p(\theta \mid x)\mathrm{d}x.
$$

Therefore, to estimate the parameter, posterior $p(\theta\mid x)$ is needed, which could be decomposed into a prior and a likelihood according to Bayes's theorem. These are what we usually need for such estimation.

## Tweedie Estimator
Here we suppose that the likelihood $p(x\mid \theta)$ conforms to a normal distribution, i.e., 

$$
p(x\mid \theta)=\mathcal{N}(\mu, \sigma^2), 
$$

and the marginal distribution $p(x)$ writes as follows 

$$
p(x)=\int^{\infty}_{-\infty}p(x\mid \theta)p(\theta)\mathrm{d}\theta.
$$

So we can derive

$$
\begin{aligned}
& \mathbb{E}[\theta \mid x]=\int_{-\infty}^{\infty} \theta p(\theta \mid x) \mathrm{d} \theta \\
& =\int_{-\infty}^{\infty} \theta \frac{p(x \mid \theta) p(\theta)}{p(x)} \mathrm{d} \theta \\
& =\frac{\int_{-\infty}^{\infty} \theta p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty} \theta \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty}\left[\sigma^2 \frac{\theta-x}{\sigma^2} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta)+x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta)\right] \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty} \sigma^2 \frac{\theta-x}{\sigma^2} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \int_{-\infty}^{\infty} \frac{\mathrm{d}\left[\frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}}\right]}{\mathrm{d} x} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \int_{-\infty}^{\infty} \frac{\mathrm{d} p(x \mid \theta)}{\mathrm{d} x} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \frac{\mathrm{d}}{\mathrm{d} x} \int_{-\infty}^{\infty} p(x \mid \theta) p(\theta) \mathrm{d} \theta+x \int_{-\infty}^{\infty} p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \frac{\mathrm{d} p(x)}{\mathrm{d} x}+x p(x)}{p(x)} \\
& =x+\sigma^2 \frac{\mathrm{d}}{\mathrm{d} x} \log p(x) \\
&
\end{aligned}
$$

Hitherto, it's obvious that the posterior expectation bears no relevance to its prior distribution, and is merely related to the marginal distribution $p(x)$. So here we arrive at the Tweedie estimator

$$
\hat{\theta}^{\text{TE}}=x+\sigma^2\frac{\mathrm{d}}{\mathrm{d}x}\log p(x)
$$


## Score Matching
For an arbitrary distribution, there is a general form to represent it originating from energy-based model[^1].

$$
p_{\theta}(x)=\frac{e^{-f_{\theta}(x)}}{Z(\theta)}, 
$$

where $Z(\theta)=\int e^{-f_{\theta}(x)}\mathrm{d}x$ is a constant for normalization, ensuring integral of the PDF $\int p_{\theta}(x)\mathrm{d}x=1$. For complex distributions, $Z(\theta)$ could be intractable, to circumvent which derivative is adopted instead.

$$
\begin{aligned}
\nabla_x \log p_{\theta}(x) &= \nabla_x \log e^{-f_{\theta}(x)}-\nabla_x \log Z(\theta)\\
&=-\nabla_x f_{\theta}(x)\\
&\approx s_{\theta}(x),
\end{aligned}
$$

i.e., approximate the gradient of logarithm of the distribution by a neural network $s_{\theta}(x)$ with parameter $\theta$. $Z(\theta)$ has been zeroed out via gradient, thus leaving no effect on our estimation. 

**Score function** is defined as the gradient of the logarithm of a distribution. And now we can perform optimization for a solution of targeted distribution

$$
L=\mathbb{E}_{p(x)}[\|s_{\theta}(x)-\nabla_x \log p(x)\|_2^2]
$$

Generally, this is what **score matching** refers to: approximating the gradient of ground truth data distribution via matching that of predicted distribution to it. 

To optimize the L2 loss above, gradient of the targeted distribution $p(x)$ with regard to data $x$ is needed. But usually we don't know the true distribution, or it's too sparse for calculating derivatives, therefore unable to supervise this optimization. We may need some further derivation.

$$
\begin{aligned}
L&=\mathbb{E}_{p(x)}[\|s_{\theta}(x)-\nabla_x \log p(x)\|_2^2]\\
&=\int p\left[\|\nabla_x \log p(x)\|^2-2\cdot \nabla_x\log p^T(x)\cdot s_{\theta}(x)+\|s_{\theta}(x)\|^2\right]\mathrm{d}x.
\end{aligned}
$$

Since $L$ is a function of $\theta$, which is what we want to optimize, it's apparent that the first term could be regarded as a constant and thus ignored. Now move on to the second term.

$$
L_2=\int p(x)\nabla_x\log p^T(x)\cdot s_{\theta}(x)\mathrm{d}x.
$$

The 2 multiplied terms share the same number of dimensions. So obviously $L_2$ is a scalar, and since $x^T \cdot y=\sum_i x_iy_i$, we have

$$
\begin{aligned}
L_2&=\int p(x)\sum\limits_i\nabla_{x_i}\log p^T(x)\cdot s_{\theta}(x)\mathrm{d}x\\
&=\sum\limits_i\int p(x)\nabla_{x_i}\log p^T(x)\cdot s_{\theta}(x)\mathrm{d}x\\
&=\sum\limits_i\int p(x)\frac{\nabla_{x_i}p^T(x)}{p^T(x)}\cdot s_{\theta}(x)\mathrm{d}x\\
&=\sum\limits_i\int\nabla_{x_i}p^T(x)\cdot s_{\theta}(x)\mathrm{d}x\\
&=\sum\limits_i\int s_{\theta}(x)\mathrm{d}p(x)\\
&=\sum\limits_i(s_{\theta}(x)\cdot p(x)-\int p(x)\mathrm{d}s_{\theta}(x)).
\end{aligned}
$$

Note that for a PDF $p(x)$, when $x\rightarrow \infty$, $p(x)\rightarrow 0$. So the first term in the deduction above can be zeroed out, and then we have

$$
\begin{aligned}
L_2&=\sum\limits_i\left(-\int p(x)\mathrm{d}s_{\theta}(x)\right)\\
&=-\sum\limits_i\int p(x)\nabla^2_{x_i}\log p_{\theta}(x)\mathrm{d}x.
\end{aligned}
$$

So we need the second derivative for each i-th element, and $\sum_i$ here is apparently corresponds to the sum of diagonal of Hessian matrix. Note that the second derivative here equals to first derivative of our neural network, i.e., 

$$
\nabla^2_{x_i}\log p_{\theta}(x)=\nabla_x s_{\theta}(x).
$$

To sum up, we arrive at the loss $L$ for score matching in its well-known form

$$
L=\mathbb{E}_{p(x)}\left[\text{tr}(\nabla_x s_{\theta}(x))+\|s_{\theta}(x)\|^2_2\right].
$$

Thus far, it's clear that knowledge about the ground truth data distribution $p(x)$ is no longer necessary when conducting score matching.

## Langevin Dynamics


# References

[^1]: LeCun, Yann, et al. "Energy-based models in document recognition and computer vision." Ninth International Conference on Document Analysis and Recognition (ICDAR 2007). Vol. 1. IEEE, 2007.