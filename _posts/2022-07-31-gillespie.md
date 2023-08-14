---
title: 'Gillespie Algorithm'
date: 2022-07-31
permalink: /posts/2023/07/gillespie/
tags:
  - stoachstic process
  - probability theory
  - MCMC
---

In probability theory, the Gillespie algorithm generates a statistically correct trajectory (possible solution) of a stochastic equation system for which the reaction rates are known. Mathematically, it is a variant of a dynamic Monte Carlo method and similar to the kinetic Monte Carlo methods.

- [Gillespie Algorithm](#gillespie-algorithm)
  - [概览](#概览)
  - [算法](#算法)
    - [思想](#思想)
    - [步骤](#步骤)
    - [相关方程](#相关方程)


# Gillespie Algorithm

## 概览

> **Time evolution of the model has been studied by computersimulation with the Gillespie algorithm (direct method), taking into account gene replication and cell division events.** *Kierzek A, Zhou L, Wanner B,* *Stochastic Kinetics Model of TCS*
> 

> **However, an equation-free analysis of the stochastic model, using the Gillespie algorithm with tau-leaping as a black-box time-stepper, in order to find stationary states for the mean of an ensemble of stochastic trajectories, reveals the long-term persistence of an approximate basal solution that combines with the graded response to produce the mixed mode.** *Hoyle RB, Avitabile D, Kierzek AM (2012) Equation-Free Analysis of Two-Component System Signalling Model Reveals the Emergence of Co-Existing Phenotypes in the Absence of Multistationarity.*
> 

<aside>
💡 **Why stochastic?**

</aside>

对于生物体内的化学反应来说，反应物分子远小于普通化学反应，速度也很慢，所以发生反应的随机性就很明显。这些随机性是由两方面造成的，一方面是因为反应物碰撞后才可能发生反应，而当分子数很少时，碰撞概率也很小；另一方面是热力学涨落，即使反应物发生碰撞，也需要有足够大的活化能才能发生相应的反应，而活化能是受热涨落影响的，具有显著的随机性。

ref: [哪位大神能详细解释一下Gillespie algorithm 和chemical system？](https://www.zhihu.com/question/45400055)

## 算法

### 思想

若时刻$t$系统状态为$x$，下一次反应在$(t+\tau)$时刻发生，而且所发生的反应是第$\mu$个反应通道$R_{\mu}$，则系统状态在$(t, \tau)$时间区间内为$x$，而$(t+\tau)$时刻变为$x+v_{\mu}$，因此根据当前时刻状态$X(t)=x$计算出下一个反应发生的时间$(t+\tau)$和相应的反应通道$R_{\mu}$，就得到了系统状态随时间的变化，即一个样本轨道。

### 步骤

1. 初始化$X(0)=x_0$，并令初始时间$t_0=0$
2. 计算$a_v=a_v(x)(v=1, \cdots, M)$，并令$a_0=\sum_{v=1}^Ma_v$
3. 产生一组随机数$(\tau, \mu)$，使其分布满足概率密度函数$P(\tau, \mu;x)$: 给定$t$时刻系统状态$X(t)=x$，系统中下一个反应在无穷小时间区间$d\tau$内发生，且发生的是反应通道$R_{\mu}$的概率
4. 令$t=t+\tau$，并根据反应通道$R_{\mu}$更新分子数目$X_i\rightarrow X_i+v_{ui}$
5. 重复2~4步

ref: [12_stochastic_simulation](http://www.be150.caltech.edu/2019/handouts/12_stochastic_simulation.html)

### 相关方程

$$
P(\tau, \mu ; x)=P_{0}(\tau, x) \times a_{\mu}(x) d \tau
$$

$$
\frac{\partial P_{0}\left(\tau^{\prime}, x\right)}{\partial \tau^{\prime}}=-\sum_{v=1}^{M} a_{v}(x) P_{0}\left(\tau^{\prime}, x\right) P_{0}(0, x)=1
$$

$$
P_{0}(\tau, \mu ; x)=\left\{\begin{array}{l}a_{\mu}(x) e^{-a_{0}(x) \tau}, \text { if } 0 \leq \tau<\infty \text { and } \mu=1, \ldots, M \\0, \text { otherwise }\end{array}\right.
$$