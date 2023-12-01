## NPA: Improving Large-scale Graph Neural Networks with Non-parametric Attention

### Overview

This repo contains an example implementation of the Non-Parameter Attention (NPA) with SGC as a baseline. Equipped with NPA, SGC can achieve significant performance gains without losing scalability.


### Requirements
Environments: Win-64 or Linux-64, GPU:TITAN RTX. Other requirements can be seen in `requirements.txt`.

Note that we offer a pure python version and a python+cpp version to accelerate python loop. The details can be seen in `run.sh`.


### Run the Codes
To evalute the performance of SGC+NPA, please run the commands in `run.sh`.


### Test Results
| Method | Cora | Citeseer | Pubmed |
| :---: | :----: | :---: | :---: |
| SGC | 81.0 | 71.9 | 78.9 |
| SGC+NPA | 83.0 | 73.6 | 80.1 |





### Theoretical Analysis

**Theorem 1:** In discussed non-parametric GNNs, The classifier $\Phi(\mathbf{x})$ has 2 properties:

- Property 1: Given feature $\mathbf{x_a} \in R^f$ ($f$ is feature dimension) and a non-zero scalar $m \in R$, $\arg\max \Phi(\mathbf{x_a}) = \mathop{\arg\max}\Phi(m\mathbf{x_a})$.

- Property 2: Given two features $\mathbf{x_a}, \mathbf{x_b} \in R^f$, if $\arg \max \Phi(\mathbf{x_a}) = \arg\max(\mathbf{x_b})$, then $\mathop{\arg\max}\Phi(\mathbf{x_a} + \mathbf{x_b}) = \arg\max\Phi(\mathbf{x_a}) = \arg\max\Phi(\mathbf{x_b})$.

**Proof.** When $\Phi$ is a linear classifier, i.e., $\Phi(\mathbf{x}) = softmax(\mathbf{Wx})$, due to the property of linear transformation and softmax, Property 1 and Property 2 are held. When $\Phi$ is a Multi-Layer Perceptron, i.e., $\Phi(\mathbf{x}) = softmax(MLP(\mathbf{x}))$. Assume that the activation in MLP is ReLU. We decompose the proof into each layer in MLP. Each layer contains a linear transformation and ReLU. Since ReLU is monotonic increasing when $x \geq 0$, it will not distort the relative relationship of two scalars. Due to the property of linear transformation and ReLU, Property 1 and Property 2 are held in each hidden layer. In the output layer, softmax will not distort the relative relationship of two scalars, either. Thus Property 1 and Property 2 are also held in the final output.

We use $\mathbf{x_i}$ to denote the feature of node $v_i$ and use $y_i$ to denote the label of node $v_i$. Assume that the graph is ideally homogeneous, i.e., $\forall v_j \in N(v_i), y_j = y_i$. In other words, if we have a well-trained classifier $\Phi(\mathbf{x})$, we have:


$$
\forall v_j \in N(v_i), \mathop{\arg\max} \Phi(\mathbf{x_i}) = \mathop{\arg\max}\Phi(\mathbf{x_j}). \tag{1} \label{eq: ideal homogeneous}
$$


Considering the node $v_i$ and its immediate neighbors $v_j \in N(v_i)$ and the corresponding $w_{ij}$, we assume that each neighbor's feature is close to the $x_i$ with error $\mathbf{\epsilon_{ij}}$ :


$$
\mathbf{x_j} = \mathbf{x_i} + \mathbf{\epsilon_{ij}},
$$


thus with weights $w_{ij}, \sum_{v_j \in N(v_i)}w_{ij} = 1$, the node's feature $x_i$ can be reconstructed by its immediate neighbors with errors:


$$
\mathbf{x_i} = \sum_{v_j \in N(v_i)} w_{ij}\mathbf{x_j} - \sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}}. \tag{2} \label{eq: reconstruction}
$$


Here we sort the edge weights $w_{ij}$ in the descendant order, i.e., $\{w_{ij}\} = \{w_{ij_1} > w_{ij_2} > ... > w_{ij_{|N(v_i)|}}\}$. And rearrange its immediate neighbors' features such that the $L_2$ norm of $\epsilon_{ij}$ is ascendant, i.e., $\{\mathbf{x_j}\} = \{\mathbf{x_{j_1}}, ..., \mathbf{x_{j_k}}, \mathbf{x_{j_{k+1}}}, ..., \mathbf{x_{j_{|N(v_i)|}}}\}, \forall k \in [1, |N(v_i)|-1], \|\mathbf{\epsilon_{ij_k}}\|_2 < \|\mathbf{\epsilon_{ij_{k+1}}}\|_2$, we have:

**Theorem 2:** Assigning weight $w_{ij_k}$ to neighbor feature $\mathbf{x_{j_k}}$ in propagation can lead to learning a well-trained classifier easier.

**Proof.** Since we assume that we have an ideal homogeneous graph, from the perspective of Eq. $\eqref{eq: ideal homogeneous}$ of the homogeneous graph, Property 1 and Property 2 of well-trained classifier $\Phi$ in Theorem 1, we have:


$$
\begin{equation}
\begin{split}
\mathop{\arg\max} \Phi(\mathbf{x_i})
=& \mathop{\arg\max} \Phi(\mathbf{x_j}; v_j \in N(v_i)) \\
=& \mathop{\arg\max} \Phi(w_{ij}\mathbf{x_j}; v_j \in N(v_i)) \\
=& \mathop{\arg\max} \Phi(w_{ij}\mathbf{x_j}+w_{ik}\mathbf{x_k}; v_j, v_k\in N(v_i))\\
=& \mathop{\arg\max} \Phi(\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j}).
\end{split}
\end{equation}
$$


From another perspective, with Eq. $\eqref{eq: reconstruction}$, we have:


$$
\begin{equation}
\small
\mathop{\arg\max} \Phi(\mathbf{x_i}) = \mathop{\arg\max} \Phi(\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j} - \sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}}).
\end{equation}
$$

Since both prediction $\mathop{\arg\max} \Phi(\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j})$ and $\mathop{\arg\max} \Phi(\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j} - \sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}})$ are equal to the prediction $\mathop{\arg\max} \Phi(\mathbf{x_i})$, a expected classifier should make the same prediction given the two inputs. If the two inputs are closer, the classifier is much easier to train to make the same prediction. On the contrary, if the two inputs are far away, the classifier must approximate a complex decision manifold in the feature space, which could be intractable, leading to sub-optimal results. Thus, we measure the difference between $\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j}$ and  $\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j} - \sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}}$ as:
















$$
\begin{equation}
\begin{split}
diff. &= \|(\sum_{v_j \in N(v_i)}w_{ij}\mathbf{x_j}) - (\sum_{v_j \in N(v_i)} w_{ij}\mathbf{x_j} - \sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}})\|_2  \\
&= \|\sum_{v_j \in N(v_i)} w_{ij}\mathbf{\epsilon_{ij}}\|_2 \\
&\leq \sum_{v_j\in N(v_i)}w_{ij}\|\mathbf{\epsilon_{ij}}\|_2 ,
\end{split}
\end{equation}
$$

















and according to the *rearrangement inequality*, letting $\{w_{ij}\}$ be the reversed order of $\{\|\mathbf{\epsilon_{ij}\|_2}\}$ can minimize the upper bound of the difference, which may further reduce the difference between two inputs. To conclude, we showed that re-assigning propagation weights according to "how close between the neighbor node feature and its center node feature" is beneficial in achieving optimal prediction performance. From this perspective, our local attention is designed to additionally assign more weights to the nodes that have more similar features with that of their center nodes.

