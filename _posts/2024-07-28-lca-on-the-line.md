---
title: 'LCA-on-the-Line Explained'
date: 2024-07-28
permalink: /posts/2024/07/lca-on-the-line/
tags:
  - generalization
  - deep learning
---

Using in-distribution (ID) accuracy to analyze out-of-distribution (OOD) performance of models can encounter severe disintegrity, especially for models trained with diverse supervision and distributions, e.g., vision models (VM) and vision-language models (VLM). This paper, accepted at ICML 2024[^1], proposed a novel, yet simple approach to measure OOD performance without introduction of any OOD data. 

- [LCA Distance](#lca-distance)
- [Using LCA Distance to Enhance Generalization](#using-lca-distance-to-enhance-generalization)


# LCA Distance

LCA, or lowest common ancestor, is a concept in graph theory (particularly for tree - a special type of DAG). When I first look at the title and the abstract, what I am most curious about is how the graph/tree is constructed - you need a graph/tree first to find the LCA for an arbitrary pair of nodes. In this paper, the authors first employ existing large-scale lexical database like WordNet as the class taxonomy, and then propose to construct such hierarchies using K-means based on the similarities of the semantic information of class labels. 

With that in mind, we can simply regard the LCA Distance as how close to the ground truth the prediction can be in a way that is somewhat knowledge-graph-based. And Figure 1 of this paper depicts how LCA distance, as a simple, robust measurement, excel in measuring OOD performance and outweigh conventional approaches where ID accuracy is adopted. 

Overall, the LCA distance is described as 

$$
D_{\text{LCA}}(y^\prime, y)=f(y)-f(N_{\text{LCA}}(y^\prime, y)), 
$$

where $N_{\text{LCA}}(y^\prime, y)$ denotes the LCA node (a class label) for predicted class $y^\prime$ and ground truth $y$. $f(\cdot)$ is a function on nodes, such as the tree depth or entropy. 

# Using LCA Distance to Enhance Generalization

With aforementioned definition of LCA distance, it's safe now to design respective loss functions to guide the optimization so that the model can be trained in a more generalized way. For a dataset with $n$ classes, we first establish an $n \times n$ LCA distance matrix $\boldsymbol{M}$, where $M_{ik}$ denotes the pairwise LCA distance of class nodes $i$ and $k$. Then the matrix is scaled by a temperature term T, and min-max scaling is applied to normalize the values between 0 and 1. 

$$
\boldsymbol{M}_{\text{LCA}}=MinMax(\boldsymbol{M}^T).
$$

Now cross entropy (CE) or binary cross entropy (BCE) losses can be employed, and in the original paper, the authors use a weighted combination of ID CE and LCA CE to make it robust in both scenarios, that is, 

$$
\mathcal{L}=\lambda\mathcal{L}_{\text{CE}}+\mathcal{L}_{\text{CE}_{\text{LCA}}}. 
$$

[^1]: For the original paper/project page, see [https://elvishelvis.github.io/papers/lca/](https://elvishelvis.github.io/papers/lca/)