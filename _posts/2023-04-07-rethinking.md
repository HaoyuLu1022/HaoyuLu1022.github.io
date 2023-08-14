---
title: 'Reflections on Researches'
date: 2023-04-07
permalink: /posts/2023/04/rethinking/
tags:
  - reflection
---

A general overview of papers read recently, and a reflection on what we have done and what to do next.

- [端到端的医学影像重建 (Compton/PET)](#端到端的医学影像重建-comptonpet)
	- [GAN](#gan)
	- [Flow](#flow)
	- [UNet](#unet)
	- [问题](#问题)
- [Object Localization](#object-localization)
	- [WSOL: 弱监督目标定位](#wsol-弱监督目标定位)
		- [什么是WSOL](#什么是wsol)
		- [图像级标注 image-level annotation](#图像级标注-image-level-annotation)
		- [通常思路: CAM](#通常思路-cam)
		- [CI-CAM](#ci-cam)
- [Causal Inference: 因果推断](#causal-inference-因果推断)
- [GCN 图卷积神经网络](#gcn-图卷积神经网络)
	- [相关应用](#相关应用)
		- [Conformation Generation](#conformation-generation)
		- [GraphNet](#graphnet)
		- [Point-GNN](#point-gnn)
		- [Track Reconstruction](#track-reconstruction)
		- [Interaction Network](#interaction-network)
	- [数据集组织](#数据集组织)
	- [To Start with: GraphSAGE](#to-start-with-graphsage)


# 端到端的医学影像重建 (Compton/PET)
## GAN
- Shen, G., Dwivedi, K., Majima, K., Horikawa, T., & Kamitani, Y. (2019). End-to-end deep image reconstruction from human brain activity. _Frontiers in computational neuroscience_, _13_, 21
## Flow
- RealNVP: Hajij, M., Zamzmi, G., Paul, R., & Thukar, L. (2022, March). Normalizing Flow for Synthetic Medical Images Generation. In _2022 IEEE Healthcare Innovations and Point of Care Technologies (HI-POCT)_ (pp. 46-49). IEEE.
## UNet
- Zhao, R., Zhang, Y., Yaman, B., Lungren, M. P., & Hansen, M. S. (2021). End-to-end AI-based MRI reconstruction and lesion detection pipeline for evaluation of deep learning image reconstruction. _arXiv preprint arXiv:2109.11524_.
## 问题
- 多数end-to-end是从sinogram (projection domain) 映射变换到重建图像 (image domain)，而Compton没有sinogram，且sinogram的引入会带来误差
- 受限于ground truth成像质量，ground truth成像质量即其理论上界；产生高质量的ground truth、标注数据也会带来较大的时间代价
- 如果辐射源不是点，而是面，那么这种条件下的“定位”，似乎跟图像重建没有区别
- 如果从list mode数据出发，模型重建图像的可解释性？
---
# Object Localization
## WSOL: 弱监督目标定位
### 什么是WSOL
从image-level annotation出发，对目标进行定位
### 图像级标注 image-level annotation
仅标注图像中相关物体所属的类别，是最简单的标注
![](https://pic3.zhimg.com/80/v2-4d6954d055271e5993e5dafb1fc99fce_720w.webp)
基于图像级标注，WSOL旨在给出带预测图像中包含主体的类别以及位置（bounding box）
### 通常思路: CAM
CAM，即Class Activation Map。通常模型首先生成CAM，然后分割出激活值最高的区域作为一个粗糙的定位结果。
### CI-CAM
[[CI-CAM]]是基于因果推断的CAM模型。主要解决在定位问题中，受到背景信息$C$纠缠从而使输入$X$和目标$Y$之间存在虚假相关性的问题
![image.png](https://raw.githubusercontent.com/HalveLuve/Images/master/PicGo/20230411181938.png)

> Shao, F., Luo, Y., Zhang, L., Ye, L., Tang, S., Yang, Y., & Xiao, J. (2021, October). Improving Weakly Supervised Object Localization via Causal Intervention. In _Proceedings of the 29th ACM International Conference on Multimedia_ (pp. 3321-3329).
---
# Causal Inference: 因果推断
> 待补充
---
# GCN 图卷积神经网络
对每个事件，absorber和scatterer上的两个响应位置，再加上source location可以构成子图。事件累积，也即子图累积，最终形成一个图。理想情况下，该图具有以下特点
- source location是一个关键点，连接所有子图
- 该图不同于一般的抽象图，具有固定的几何构型（边长+角度），不可变形
	- 所以理论上图同构没有意义
- 单向图：只有scatter指向absorber的边
- 重边：相同的一组scatter和absorber可能重复形成event

## 相关应用
### Conformation Generation
预测分子的可能坐标，即Conformation Generation，常用于蛋白质结构预测。
> Mansimov, E., Mahmood, O., Kang, S. _et al._ Molecular Geometry Prediction using a Deep Generative Graph Neural Network. _Sci Rep_ **9**, 20381 (2019). https://doi.org/10.1038/s41598-019-56773-5
### GraphNet
一个用于中微子望远镜图像重建的开源框架
> https://github.com/graphnet-team/graphnet
> 没看懂，至少说明核物理也存在这种思路的应用
### Point-GNN
对3D点云进行目标检测
> Shi, W., & Rajkumar, R. (2020). Point-gnn: Graph neural network for 3d object detection in a point cloud. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ (pp. 1711-1719).
> https://github.com/WeijingShi/Point-GNN
### Track Reconstruction
高能物理实验中的粒子轨迹追踪/重建
> Ju, X., Farrell, S., Calafiura, P., Murnane, D., Gray, L., Klijnsma, T., ... & Usher, T. (2020). Graph neural networks for particle reconstruction in high energy physics detectors. _arXiv preprint arXiv:2003.11603_.
> https://jduarte.physics.ucsd.edu/phys139_239/finalprojects/Group3_Report.pdf
### Interaction Network
- The first general-purpose, learnable physics engine, and a powerful general framework for reasoning about object and relations in a wide variety of complex real-world domains
- takes graphs as input, performs object- and relation-centric reasoning in a way that is analogous to a simulation, and is implemented using deep neural networks
就是一个针对物理问题的传统GCN模型，其核心也是Message Passing；但是原文以及多处应用均提到了它对于解决不同领域物理问题的**泛用性**
> Battaglia, P., Pascanu, R., Lai, M., & Jimenez Rezende, D. (2016). Interaction networks for learning about objects, relations and physics. _Advances in neural information processing systems_, _29_.
## 数据集组织
如果不考虑event的先后顺序
- 建立邻接矩阵$E_{adj}[21][21][3]$：每个邻接矩阵放21个events的feature
	- `axis=0`：absorber对应的scatterer
	- `axis=1`：$E_{1}$
	- `axis=2`：$E_{2}$
- 保留响应点的几何特征：$(x_{1}, y_{1}, z_{1}), (x_{2}, y_{2}, z_{2})$
> 可以通过预测需要的邻接矩阵个数来判断数据量上的开销
> 是否同时需要$E_{1}$和$E_{2}$有待测试；可以只保留保留$E_{1}$，或者另一个换成$511-E_{1}$
## To Start with: GraphSAGE
GraphSAGE针对之前的网络表示学习的transductive的问题，提出了一个inductive的GraphSAGE算法；同时利用节点特征信息和结构信息得到Graph Embedding的映射。相比之前的方法，之前都是保存了映射后的结果，而GraphSAGE保存了生成embedding的映射，可扩展性更强，对于节点分类和链接预测问题的表现也比较突出

---
