---
title: 'Radon Transform'
date: 2023-06-29
permalink: /posts/2023/06/radon-transform/
tags:
  - Radon transform
---

The Radon transform is the transform of our n-dimensional volume to a complete set of (n-1)-dimensional line integrals. Whereas the inverse Radon transform is the transform from our complete (n-1)-dimensional line integrals back to the original image.

- [Notations](#notations)
- [Definition](#definition)
  - [$\\delta$函数](#delta函数)
  - [直线表示](#直线表示)
- [Radon逆变换](#radon逆变换)
  - [Fourier变换：卷积性质](#fourier变换卷积性质)
- [Funk transform](#funk-transform)


# Notations
- $f(x, y)$：一个二维函数，表示被检体的吸收率（或称为物质密度）
# Definition
直线$L$穿过$f(x, y)$，所对应的强度即函数$f(x, y)$在直线$L$上的线积分
$$
\mathcal{R}_{L}=\int_{L}f(x, y)ds
$$
## $\delta$函数
$$
\delta(x-x_{0})= 
\begin{cases}
\infty,\ x=x_{0}\\
0,\ x\ne x_{0}
\end{cases}
$$
且
$$
\int_{-\infty}^{+\infty}\delta(x)dx=1
$$
## 直线表示
直线$L$远离原点的法线方向为$\vec{n}=(\cos{\theta}, \sin{\theta})$，原点到直线的距离为$p$。则直线表示为
$$
x\cos{\theta}+y\cos{\theta}=p
$$
方向由$\theta$唯一定义，该方向上的每一点放射源由$p$唯一定义。则根据$(\theta, p)$可以定义Radon变换
$$
\begin{aligned}
\mathcal{R}_{L}&=\int_{L}f(x, y)ds\\
\mathcal{R}(\theta, p)&=\int_{(\theta, p)}f(x, y)ds\\
\mathcal{R}(\theta, p)&=\int\limits_{x\cos{\theta}+y\cos{\theta}-p}f(x, y)ds\\
\mathcal{R}(\theta, p)&=\iint f(x, y)\cdot \delta(p-x\cos{\theta}-y\sin{\theta})dxdy
\end{aligned}
$$
# Radon逆变换
根据不同的$(\theta, p)$求出$f(x, y)$。在积分形式下得出函数表达式，第一想到的是Fourier变换
## Fourier变换：卷积性质
对
$$
f(t)\star g(t)=\int f(\tau)g(t-\tau)d\tau
$$
Fourier变换得
$$
\begin{aligned}
\mathcal{F}[f(t)\star g(t)]&=\int\left[\int f(\tau)g(t-\tau)d\tau\right]e^{-j\omega t}dt\\
&=\int\left[\int f(\tau)g(t-\tau)d\tau\right]e^{-j\omega (\tau+t-\tau)}dt\\
&=\int f(\tau)e^{-j\omega\tau}\left[\int g(t-\tau)e^{-j\omega (t-\tau)}dt\right]d\tau\\
&=\int f(\tau)e^{-j\omega\tau}\hat{g}(\omega)d\tau\\
&=\hat{f}(\omega)\hat{g}(\omega)
\end{aligned}
$$
类似地，对$\mathcal{R}_{(\theta,p)}$代入Fourier变换得
$$
\begin{aligned}
\mathcal{F}[\mathcal{R}(\theta, p)] & =\int\left[\iint f(x, y) \delta(p-x \cos \theta-y \sin \theta) \mathrm{d} x \mathrm{~d} y\right] e^{-j w p} \mathrm{~d} p \\
& =\int\left[\iint f(x, y) \delta(p-x \cos \theta-y \sin \theta) \mathrm{d} x \mathrm{~d} y\right] e^{-j w(p-x \cos \theta-y \sin \theta + x\cos \theta+y\sin \theta) } \mathrm{~d} p  \\
& =\iint f(x, y) e^{-j w(x \cos \theta+y \sin \theta)}\left[\int \delta(p-x \cos \theta-y \sin \theta) e^{-j w(p-x \cos \theta-y \sin \theta)}\mathrm{~d} p \right]\mathrm{~d} x \mathrm{~d} y  \\
& =\iint f(x, y) e^{-j w(x \cos \theta+y \sin \theta)} \hat{\delta}(w) \mathrm{d} x \mathrm{~d} y \\
& =\hat{\delta}(w) \iint f(x, y) e^{-j w(x \cos \theta+y \sin \theta)} \mathrm{d} x \mathrm{~d} y \\
& =\hat{f}(w \cos \theta, w \sin \theta) \hat{\delta}(w)
\end{aligned}
$$
$\delta$函数的Fourier函数变换为常数1，故最终的等式为
$$
\hat{f}(\omega \cos \theta, \omega\sin\theta)=\mathcal{F}[\mathcal{R}(\theta, p)]
$$
# Funk transform
在球的中心圆上对函数进行积分。令$f$为$\mathbb{S}^2$上的连续函数，那么对于单位向量$\mathbf{x}$，有
$$
\mathcal{F}f(\mathbf{x})=\int\limits_{\mathbf{u}\in C(x)}f(\mathbf{u})\mathrm{~d}s(\mathbf{u})
$$
其中积分是在球的中心圆$C(x)$的弧长$\mathrm{~d}s$上进行的。球的中心圆$C(x)$包含所有垂直于$\mathbf{x}$的单位向量
$$
C(x)=\{\mathbf{u}\in \mathbb{S}^{2}|\mathbf{u}\cdot\mathbf{x}=0\}
$$