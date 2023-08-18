---
title: 'Geometric Graph'
date: 2023-04-01
permalink: /posts/2023/04/geometric-graph/
tags:
  - geometry
  - graph theory
  - GNN
---

A graph is a powerful way of representing relationships among entities as nodes connected by edges. Sometimes nodes and edges can have spatial features, such as 3D coordinates of nodes and directions along edges. How do we reason over the topology of graphs while considering those geometric features?

- [Concepts](#concepts)
	- [Geometry](#geometry)
	- [Graph](#graph)
	- [Geometric Deep Learning](#geometric-deep-learning)
- [GVP: Geometric Vector Perceptron](#gvp-geometric-vector-perceptron)
	- [如何处理](#如何处理)
- [GVP-GNN](#gvp-gnn)
	- [GNN Basics](#gnn-basics)
		- [Message Function](#message-function)
		- [Update Function](#update-function)
	- [GVP-GNN: Message from Vector Features](#gvp-gnn-message-from-vector-features)
		- [Message Function](#message-function-1)
		- [Update Function](#update-function-1)


# Concepts
## Geometry
> 我们的问题处于欧氏几何 (Euclidean) 范畴内

狭义而言，几何通常约定俗成地被理解为欧式几何；但是还有其他几何存在，如Riemann几何
为了统一，Erlangen纲领将几何视作对多种变换的不变性的探究。

## Graph
A graph is a common data structure using nodes connected by edges to represent complex relationships among entities. Generally, 

$$
G = (V, E)
$$

- 某些情况下，结点和边可能具有空间上的特征，例如结点的3D坐标、边的方向等等。我们称这种graph为**Geometric Graph**
- 无论是朴素的graph还是Geometric Graph，其结点都满足**不变性 (invariance)**：结点的顺序不影响结点的属性，如度、中心度；所以广义的graph都属于geometric object
- Geometric Graph满足对欧式变换 (旋转、平移、对称)的不变性


## Geometric Deep Learning

通过Erlangen纲领的对称性和不变性，将多种深度学习 (如CNN、GNN) 的方法统一起来。对于任意grometric object，GDL的设计范式包含以下3个要素
- local equivariant map
- global invariant map
- coarsening operator


# GVP: Geometric Vector Perceptron

顾名思义，跟其他神经网络中的全连接层使用的一样，GVP也是一种神经元。其新颖之处在于可以一起处理来自两个不同层面的特征：
- Scalar feature $\textbf{s}\in \mathbb{R}^{h}$
- Geometric vector feature $\textbf{V}\in \mathbb{R}^{v\times 3}$


## 如何处理

与空间信息无关的特征向量。宽泛而言，一般的神经元对$\textbf{s}$的变换如下

$$
\mathbf{s}^{\prime}=\operatorname{Perceptron}(\mathbf{s})=\sigma\left(\mathbf{w}^{\top} \mathbf{s}+b\right)
$$

而GVP可以同时处理特征$\textbf{s}$和$\textbf{V}$

$$
\begin{aligned}
\mathbf{s}^{\prime}, \mathbf{V}^{\prime}&=G V P(\mathbf{s}, \mathbf{V}) \\
\mathbf{s}^{\prime}&=\sigma\left(\mathbf{W}_m\left[\begin{array}{c}
\left\|\mathbf{W}_h \mathbf{V}\right\|_2 \\
\mathbf{s}
\end{array}\right]+\mathbf{b}\right) \\
\mathbf{V}^{\prime}&=\sigma^{+}\left(\left\|\mathbf{W}_\mu \mathbf{W}_h \mathbf{V}\right\|_2\right) \odot \mathbf{W}_\mu \mathbf{W}_h \mathbf{V}
\end{aligned}
$$

其中

$$\textbf{W}_{\mu},\textbf{W}_{h},\textbf{W}_{m}$$

是对特征$\textbf{s}$和$\textbf{V}$进行线性变换的权重矩阵
- 可以证明输出$\mathbf{s}^{\prime}$是$E(3)$不变的，$\mathbf{V}^{\prime}$是$E(3)$等变的
	- $\mathbf{s}^{\prime}$只与向量$\mathbf{V}$的L2范数有关，相对于欧式变换保持不变
	- $\mathbf{V}^{\prime}$与$\mathbf{W}_\mu \mathbf{W}_h \mathbf{V}$有关，只有发生欧式变换时才变化，因此等变


# GVP-GNN
## GNN Basics

GNN的重要机制是message，即通过边、点的特征，在邻接结点中交换message已更新node representation，从而在推理时整合graph的拓扑结构
- 用$\mathbf{h}_{i}$代表结点$i$的特征
- 用$\mathbf{h}_{j}$代表结点$j$的特征
- $\mathbf{e}_{ij}$代表边的特征

### Message Function

描述一对结点和其连边如何生成message $\mathbf{m}_{ij}$

$$
\mathbf{m}_{ij}=M(\mathbf{h}_{i}, \mathbf{h}_{j}, \mathbf{e}_{ij})
$$

### Update Function

来自一个结点的所有邻接结点的message被聚合在一起，用于更新该结点的node representation

$$
\mathbf{h}_i \leftarrow U\left(\mathbf{h}_i, A G G_{j \in \mathcal{N}(i)} \mathbf{m}_{i j}\right)
$$

## GVP-GNN: Message from Vector Features

GVP-GNN遵循相同的message范式，额外允许边和点同时具备scalar feature和vector feature
* 用$(\mathbf{s}_j, \mathbf{V}_j)$代表结点的特征
* 用$(\mathbf{s}_{i j}, \mathbf{V}_{i j})$代表边的特征

### Message Function

$$
\begin{aligned}
\mathbf{m}_{i j} & =M_{G V P}\left(\mathbf{s}_j, \mathbf{V}_j, \mathbf{s}_{i j}, \mathbf{V}_{i j}\right) \\
& =\operatorname{GVP}\left(\operatorname{concat}\left(\mathbf{s}_j, \mathbf{s}_{i j}\right), \operatorname{concat}\left(\mathbf{V}_j, \mathbf{V}_{i j}\right)\right)
\end{aligned}
$$

### Update Function
为了简化表达，令

$$\mathbf{h}=(\mathbf{s}, \mathbf{V}), \mathbf{m}_{ij}=(\mathbf{s}^{\prime}_{ij}, \mathbf{V}^{\prime}_{ij})$$

则有

$$
\mathbf{h}_i \leftarrow \operatorname{Layer} \operatorname{Norm}\left(\mathbf{h}_i+\frac{1}{|\mathcal{N}(i)|} \operatorname{Dropout}\left(\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{i j}\right)\right)
$$

GVP-GNN通过结合边和点的vector feature (`concat(Vj, Vij)`)，实现了对SE(3)的等变性，从而能够整合边的特征