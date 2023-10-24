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


