---
title: 'Locating Objects Without Bounding Boxes'
date: 2023-02-05
permalink: /posts/2023/02/hausdorff/
tags:
  - object localization
  - Hausdorff distance
---

This loss function is a modification of the average Hausdorff distance between two unordered sets of points. The proposed method has no notion of bounding boxes, region proposals, or sliding windows.

ref: [[1806.07564] Locating Objects Without Bounding Boxes (arxiv.org)](https://arxiv.org/abs/1806.07564)

- [Hausdorff距离](#hausdorff距离)
	- [单向Hausdorff距离](#单向hausdorff距离)
	- [双向Hausdorff距离](#双向hausdorff距离)
- [平均Haussdorff距离 AHD](#平均haussdorff距离-ahd)
	- [问题](#问题)
- [带权Haussdorff距离 WHD](#带权haussdorff距离-whd)
	- [相对于pixelwise loss的优势](#相对于pixelwise-loss的优势)


可用于任何 FCN 来估计目标位置的损失函数（weighted Hausdorff distance），从而不用任何边界框来估计一个图像中物体的位置和数量
# Hausdorff距离
描述两组点集之间相似程度的一种量度
假设有两组点集$A=\{a^{1}, a^{2}, \cdots, a^{p}\}, B=\{b^{1}, b^{2}, \cdots, b^{p}\}$
## 单向Hausdorff距离
点集$A, B$之间的单向Hausdorff距离
$$\begin{aligned}
h(A, B)&=\max\limits_{a\in A}\min\limits_{b\in B}||a-b||\\
h(B, A)&=\max\limits_{b\in B}\min\limits_{a\in A}||b-a||
\end{aligned}
$$
其中$||a-b||$表示$a$与$b$之间的距离范式，如L2或Euclidean距离
> 	$h(A, B)$：先在点集$B$中取距离点集$A$最近的点$b^{j}$，然后计算每个$a^{i}\in A$与点$b^{j}$的距离，取其中的最大值作为$h(A, B)$
> 	若$h(A, B)=d$，则$A$中所有点到$B$的距离不超过$d$
## 双向Hausdorff距离
$$
H(A, B)=\max\{h(A, B), h(B, A)\}
$$
双向Hausdorff距离取两个单向Hausdorff距离的最大值，度量两个点集的不相似程度
- 对噪声/异常值敏感
![](https://img-blog.csdnimg.cn/20200603191338464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpemhpc2h1aXhpb25n,size_16,color_FFFFFF,t_70)
# 平均Haussdorff距离 AHD
令点集$X, Y$，$X$即预测，$Y$即ground truth（像素坐标）；其平均Hausdorff距离为
$$
H_{avg}(X, Y)=\frac{1}{|X|}\sum\limits_{x\in X}\min\limits_{y\in Y}d(x, y)+\frac{1}{|Y|}\sum\limits\limits_{y\in Y}\min\limits_{x\in X}d(x, y)
$$
其中$d(\cdot, \cdot)$可以是任意度量
## 问题
- 带线性层的CNN隐式决定了预测点的个数$|X|$，即网络最后一层的size
- U-Net等FCN可以通过higher activation表现对象的中心，但不能得到像素坐标
- 为了便于后向传播学习，损失函数必须是可微的
# 带权Haussdorff距离 WHD
$$
H_{w}(p, Y)=\frac{1}{S+\epsilon}\sum\limits_{x\in \Omega}p_{x}\min\limits_{y\in Y}d(x, y)+\frac{1}{|Y|} \sum_{y \in Y} M_{x \in \Omega}^{\alpha}\left[p_x d(x, y)+\left(1-p_x\right) d_{\max }\right]
$$
其中
$$
\begin{aligned}
S&=\sum\limits_{x\in \Omega}p_{x}\\
M_{x\in A}^{\alpha}[f(x)]&=(\frac{1}{|A|}\sum\limits_{x\in A}f^{\alpha}(x))^{\frac{1}{\alpha]}}
\end{aligned}
$$
- $p_{x}\in [0, 1]$是网络在像素坐标$x$处的单值输出，注意$p_{x}$不需要归一化（$\sum\limits_{x\in \Omega}p_{x}=1$不必要）
- 广义均值$M^{\alpha}[\cdot]$在$\alpha=-\infty$时取最小值
- $\epsilon$被设为$1e-6$，在$p_{x}\approx 0 , \forall x\in \Omega$时可提供数值稳定性
- 当$p_{x}=\{0, 1\}, \alpha=-\infty, \epsilon=0$时，WHD退化为AHD
- 当$H_{w}(p, Y)\geq 0$时，全局最小值（即$H_{w}(p, Y)= 0$）对应于$p_{x}=1$若$x\in Y$，反之则对应于$p_{x}=0$
- 第一项中乘以$p_{x}$是为了惩罚在图像中附近没有ground truth点$y$的high activation；损失函数对于假正有惩罚
- 第二项中，通过式$f(\cdot):=p_x d(x, y)+\left(1-p_x\right) d_{\max }$
	- 若$p_{x_{0}}\approx 1$，则$f(\cdot)\approx d(x_{0}, y)$。这意味着点$x_{0}$像AHD中那样贡献loss
	- 若$p_{x_{0}}\approx 0, x_{0}\ne y$，则$f(\cdot)\approx d_{max}$
		- 若$\alpha=-\infty$，则点$x_{0}$不会对loss做贡献，因为最小$M_{x\in \Omega}[\cdot]$会忽略点$x_{0}$
		- 若存在另一离$y$更近、且$p_{x_{1}}>0$的点$x_{1}$。那么$M[\cdot]$会选中$x_{1}$；否则$M_{x\in \Omega}[\cdot]$较高
	- 这意味着ground truth附近的low activation会被惩罚
- $f(\cdot)$并非唯一约束$\left.f\right|_{p_x=1}=d(x, y)$且$\left.f\right|_{p_x=0}=d_{max}$的表达式
---
若input图像需要被resize以输入网络，我们可以对WHD进行归一化来处理该形变。令原图尺寸为$(S_{0}^{(1)}, S_{0}^{(2)})$，resize后图像尺寸为$(S_{r}^{(0)}, S_{r}^{( 1)})$，那么WHD中的$d(x, y)$就被替换为$d(S_{x}, S_{y})$，其中$x, y\in \Omega$且
$$
\mathbf{S}=\left(\begin{array}{cc}
S_o^{(1)} / S_r^{(1)} & 0 \\
0 & S_o^{(2)} / S_r^{(2)}
\end{array}\right)
$$
## 相对于pixelwise loss的优势
pixelwise loss不清楚两点$x\in X$和$y\in Y$有多近，除非$x=y$