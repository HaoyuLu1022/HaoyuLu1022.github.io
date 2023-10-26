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
- [Denoising Diffusion Probabilistic Models](#denoising-diffusion-probabilistic-models)
  - [Forward Process](#forward-process)
  - [Backward Process](#backward-process)
- [Score-based Generative Models](#score-based-generative-models)
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

> For non-French speaker, its pronunciation is close to /laŋˈjevɔn/.

<!-- <iframe src='https://vdn3.vzuu.com/SD/a4669c08-2340-11eb-826e-be7830fcb65c.mp4?disable_local_cache=1&bu=078babd7&c=avc.0.0&f=mp4&expiration=1698256697&auth_key=1698256697-0-0-016cb3e6d691305f46153f85f3c185e5&v=tx&pu=078babd7'></iframe> -->
<center>
<video controls="" autoplay="autoplay" name="media"><source src="/images/brownian.mp4" type="video/mp4"></video>
</center>

Proposed by Paul Langevin, **Langevin equation** describes the dynamics of molecular systems[^2]. Imagine the pollen grains on the surface of water: water molecules bumping into them from random directions, coercing them into random walk. Based on Newton's law, it's easy to write

$$
\sum F=ma.
$$

And first, there is one factor known in $\sum F$: a random force $\epsilon_t$ given by water molecules, which could be hypothesized as a normal distribution, whose mean is 0 and whose variance increases as time goes by.

Besides, friction should be included: if one pollen grain has velocity in a given direction, there will be much more water molecules towards the front of it, slowing it down by friction. Therefore, it's safe to say that the friction is inversely proportional to the grain's velocity $v$, i.e., $-\gamma v$, where $\gamma$ is a friction coefficient.

Considering a generic $F$ for any possible external forces, now we arrive at

$$
ma_t=-\gamma v_t + \epsilon_t + F
$$

This is usually called the *underdamped* form. Since pollen grains are pretty light, we may just leave out the term $ma_t$, thus arriving at the *overdamped* form

$$
0=-\gamma v_t + \epsilon_t + F
$$

Force is a gradient of energy $E$ with regard to displacement $x$, and velocity is the derivative of displacement on time. So we can rewrite the formula above as

$$
0=-\gamma \frac{\mathrm{d}x}{\mathrm{d}t} + \epsilon_t + \frac{\partial E}{\partial x}
$$

And we further discretize the time derivative and have

$$
\begin{aligned}
0&=-\gamma \frac{x_{t+\mathrm{d}t}-x_t}{\mathrm{d}t} + \epsilon_t + \frac{\partial E}{\partial x}\\
\Rightarrow x_{t+\mathrm{d}t}&=x_t + \frac{\mathrm{d}t}{\gamma}\frac{\partial E}{\partial x} + \frac{\mathrm{d}t}{\gamma}\epsilon_t
\end{aligned}
$$

Now recall how weights are updated in SGD

$$
w_{t+1}=w_t-\alpha \frac{\partial L}{\partial w_t}
$$

If we neglect the random term in Langevin dynamics, and compare the two equations, we can arrive at the following conclusions:

- displacement $x$ $\Longleftrightarrow$ weight $w$
- energy $E$ $\Longleftrightarrow$ loss $L$
- noise in Brownian motion $\Longleftrightarrow$ error of mini-batch

In this manner, it seems intuitive and reasonable that Langevin dynamics and SGD are analogous. If we combine SGD and Langevin dynamics, it's basically adding Gaussian noise to the standard SGD, avoiding local optima, known as **stochastic gradient Langevin dynamics**[^3].

$$
x_{t+1}\leftarrow x_t+\delta \nabla_x \log p(x_t)+\sqrt{2\delta}\epsilon_t, t=0, 1, \dots, T, 
$$

where $x_0$ is sampled at random from a prior distribution, $\delta$ denotes step size, and $\epsilon_t\sim \mathcal{N}(0, \mathbf{I})$ is a perturbing term, ensuring that generated samples won't collapse onto a mode, but hover around it for diversity. When $\delta\rightarrow 0$ and $T\rightarrow \infty$, the ultimate $x_t$ is just the real data.

# Denoising Diffusion Probabilistic Models

![Variational Diffusion Models](/images/vdm.png)

## Forward Process

$x_0$ is the original image, and each $x_{t-1}\rightarrow x_t$ adds noise gradually to it. Transitioning between 2 adjacent states can be modelled linearly, i.e., 

$$
x_t=a_tx_{t-1}+b_t\varepsilon_t, \quad \varepsilon_t\sim \mathcal{N}(0, \mathbf{I})
$$

Obviously, since $x_{t-1}$ has more information, $a_t$ must be an attenuation coefficient, i.e., $a_t\in (0, 1)$. For simplicity, we may as well suppose that $b_t\in (0, 1)$. And now we expand the equation above as

$$
\begin{aligned}
x_t & =a_t x_{t-1}+b_t \varepsilon_t \\
& =a_t\left(a_{t-1} x_{t-2}+b_{t-1} \varepsilon_{t-1}\right)+b_t \varepsilon_t \\
& =a_t a_{t-1} x_{t-2}+a_t b_{t-1} \varepsilon_{t-1}+b_t \varepsilon_t \\
& =\ldots \\
& =\left(a_t \ldots a_1\right) x_0+\left(a_t \ldots a_2\right) b_1 \varepsilon_1+\left(a_t \ldots a_3\right) b_2 \varepsilon_2+\cdots+a_t b_{t-1} \varepsilon_{t-1}+b_t \varepsilon_t.
\end{aligned}
$$

We can see that the besides the first term, the rest is the sum of multiple independent Gaussian noise, which is also a Gaussian distribution, whose mean is 0 and variance is the sum of squares of each coefficient $\left(a_t \ldots a_2\right)^2 b_1^2+\left(a_t \ldots a_3\right)^2 b_2^2+\cdots+a_t^2 b_{t-1}^2+b_t^2$. Therefore, it can be transformed into

$$
x_t=\left(a_t \ldots a_1\right) x_0+\sqrt{\left(a_t \ldots a_2\right)^2 b_1^2+\left(a_t \ldots a_3\right)^2 b_2^2+\cdots+a_t^2 b_{t-1}^2+b_t^2}\bar{\varepsilon}_t, \\
\bar{\varepsilon}_t\sim \mathcal{N}(0, \mathbf{I}).
$$

Furthermore, if we add up the sum of squares of the coefficients, say, 

$$
\begin{aligned}
& \left(a_t \ldots a_1\right)^2+\left(a_t \ldots a_2\right)^2 b_1^2+\left(a_t \ldots a_3\right)^2 b_2^2+\cdots+a_t^2 b_{t-1}^2+b_t^2 \\
= & \left(a_t \ldots a_2\right)^2 a_1^2+\left(a_t \ldots a_2\right)^2 b_1^2+\left(a_t \ldots a_3\right)^2 b_2^2+\cdots+a_t^2 b_{t-1}^2+b_t^2 \\
= & \left(a_t \ldots a_2\right)^2\left(a_1^2+b_1^2\right)+\left(a_t \ldots a_3\right)^2 b_2^2+\cdots+a_t^2 b_{t-1}^2+b_t^2 \\
= & \left(a_t \ldots a_3\right)^2\left(a_2^2\left(a_1^2+b_1^2\right)+b_2^2\right)+\cdots+a_t^2 b_{t-1}^2+b_t^2 \\
= & a_t^2\left(a_{t-1}^2\left(\ldots\left(a_2^2\left(a_1^2+b_1^2\right)+b_2^2\right)+\ldots\right)+b_{t-1}^2\right)+b_t^2
\end{aligned}
$$

it's easy to simplify this term to $1$ when $a_t^2+b_t^2=1$. Denote $\bar{a}_t=(a_t\dots a_1)^2$, and then the variance can be written as $1-\bar{a}_t$. Then we further simplify the forward process to

$$
x_t=\sqrt{\bar{a}_t}x_0+\sqrt{1-\bar{a}_t}\bar{\varepsilon_t}, \quad \bar{\varepsilon}_t\sim \mathcal{N}(0, \mathbf{I}).
$$

Then we rewrite Eq. (17) as follows, 

$$
x_t=\sqrt{a_t}x_0+\sqrt{1-a_t}\varepsilon_t, \quad \varepsilon_t\sim \mathcal{N}(0, \mathbf{I}), 
$$

which aligns with the form in the original paper[^4]. 

## Backward Process

# Score-based Generative Models

# References

[^1]: LeCun, Yann, et al. "Energy-based models in document recognition and computer vision." Ninth International Conference on Document Analysis and Recognition (ICDAR 2007). Vol. 1. IEEE, 2007.

[^2]: Langevin, Paul. "Sur la théorie du mouvement brownien." Compt. Rendus 146 (1908): 530-533.

[^3]: Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011.

[^4]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.