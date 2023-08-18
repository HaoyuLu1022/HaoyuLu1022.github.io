---
title: 'GraphSAGE'
date: 2023-03-31
permalink: /posts/2023/03/GraphSAGE/
tags:
  - graph theory
  - GNN
---

GraphSAGE is a general inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data

SAGE: Sample and Aggregate

- [Motif](#motif)
	- [Transductive \& Inductive](#transductive--inductive)
- [思路](#思路)
	- [大致流程](#大致流程)
- [模型](#模型)
	- [符号表](#符号表)
	- [三步走](#三步走)
	- [损失函数](#损失函数)
		- [有监督](#有监督)
		- [无监督](#无监督)


# Motif

- 当时多数graph embedding框架是直推式的 (transductive)，只能对一个固定的图生成embedding，这种transductive的方法不能对图中没有的新结点生成embedding
- GraphSAGE是一个归纳式的 (inductive) 框架，能够高效地利用结点的属性信息对新结点生成embedding


## Transductive & Inductive

统计机器学习可以分成两种
- Transductive learning，直推学习：测试集是特定、固定的样本（从此个例到彼个例）
- Inductive learning，归纳学习：测试集非特定，从多个个例归纳出普遍性，再演绎到个例

GNN中经典的GCN、DeepWalk等都是transductive learning：大多数node embedding模型都基于频谱分解/矩阵分解方法，而**矩阵分解方法本质是transductive的**
- transductive learning在处理从未见过的数据时效果不佳，因而要求图形结构的所有结点都在训练时出现，以生成node embedding；如果有新结点添加到图结构中，则需要重新训练模型
- GraphSAGE (inductive) 学习到的node embedding可以根据node的邻居关系的变化而变化

# 思路

SAGE指的是Sample and Aggregate，不是对每个顶点都训练一个单独的embedding向量，而是训练一组aggregator function
- Aggregator function学习如何从一个顶点的局部邻居聚合特征信息
- 每个function从一个顶点的不同搜索深度聚合信息
- 测试/推断时，通过学习到的aggregator function对从未见过的node生成embedding


## 大致流程

- 采样：逐步采样目标顶点的$k$阶邻
- 聚合：聚合$k$阶邻的信息，获得目标顶点的embedding
- 下游任务：利用聚合生成的embedding进行预测等下游任务


# 模型
## 符号表

- $X_{v}$：结点$v$的特征
- $h_{v}^{0}$：结点$v$的初始node embedding
- $h_{v}^{k}$：结点$v$在第$k$次迭代的node embedding
- $Z_{v}$：结点$v$经过GraphSAGE模型后最终的node embedding


## 三步走
1. 初始化：$h_{v}^{k-1}=h_{v}^{0}=x_{v}$，即将所有结点的初始node embedding设为其特征向量
2. Aggregate 聚合：将向量的集合转换为向量
	- 与一般的数据（图像、文本等）不同，图中的结点是**无序**的，所以aggregator要关于无序向量集$\{h_{u}^{k-1}, \forall u\in N(v)\}$对称，其中$N(v)$代表结点$v$的邻居结点集
	- 尝试多种aggregator
		- Mean Aggregator：对应元素取均值
		- LSTM Aggregator：但LSTM输入是有序的；可以将邻居节点的向量集打乱后输入
		- Pooling Aggregator：所有相邻结点的向量共享权重，先经过非线性全连接层，然后max-pooling
	- $k$类似于感受野，过小退化为MLP，过大会稀释结点$v$自身的feature


$$
a_{v}=f_{aggregate}(\{h_{u}|u\in N(v)\})
$$
也即
$$
h_{N(v)}^{k}\leftarrow AGGREGATE_{k}(\{h_{u}^{k-1}, \forall u\in N(v)\})
$$

1. Update 更新：对于结点$v$，基于其$k$阶邻获得聚合embedding后，使用先前表示和$k$阶聚合表示共同更新当前结点$v$
	- 该Update函数可以为任何可微函数（均值 -> 神经网络）


$$
h_{v}^{k}=f_{update}(a_{v}, h_{v}^{k-1})
$$

也即

$$
h_{v}^{k}\leftarrow \sigma(\mathbb{W}^{k}\cdot CONCAT(h_{u}^{k-1}, h_{N(v)}^{k}))
$$
![image.png](https://raw.githubusercontent.com/HalveLuve/Images/master/PicGo/20230413005928.png)
注意

- 每次迭代都有一个可学习的、单独的权重矩阵$\mathbb{W}$
- 通过除以矢量范数来标准化node embedding，以防止梯度爆炸


## 损失函数
### 有监督

针对特定任务，按惯例设计即可。如节点分类-交叉熵样式

### 无监督

$$
J_{\mathcal{G}}\left(\mathbf{z}_u\right)=-\log \left(\sigma\left(\mathbf{z}_u^{\top} \mathbf{z}_v\right)\right)-Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log \left(\sigma\left(-\mathbf{z}_u^{\top} \mathbf{z}_{v_n}\right)\right)
$$
