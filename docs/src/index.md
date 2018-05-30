# SpectralClustering.jl

Given a set of patterns $X=\{x_1,x_2,...x_n\} \in {\mathbb R}^m$, and a simmilarity function  $d:{\mathbb R}^m \times {\mathbb R}^m  \rightarrow {\mathbb R}$, is possible to build an affinity matrix $W$ such that  $W(i,j) = d(x_i, x_j)$. Spectral clustering algorithms obtains a low rank representation of the patterns solving the following optimization problem

$\begin{array}{ccc}
\max & \mbox{Tr}(U^T L  U) \\
U \in {\mathbb R}^{n\times k} & \\
\textrm{s.a.} & {U^T U}  =   I
\end{array}$

where $L = D^{-\frac{1}{2}}WD^{-\frac{1}{2}}$ is the Laplacian matrix derived from $W$ according [ng2002spectral](#ng2002spectral) and $D$ is a diagonal matrix with the sum of the rows of $W$ located in its main diagonal. Once obtained $U$, their rows are considered as the new coordinates of the patterns. In this new representation is simpler to apply a traditional clustering algorithm  [shi2000normalized](#shi2000normalized).


Spectral graph partitioning methods have been successfully
applied to circuit layout [3, 1], load balancing [4] and
image segmentation [10, 6]. As a discriminative approach,
they do not make assumptions about the global structure of
data. Instead, local evidence on how likely two data points
belong to the same class is first collected and a global decision
is then made to divide all data points into disjunct sets
according to some criterion. Often, such a criterion can be
interpreted in an embedding framework, where the grouping
relationships among data points are preserved as much
as possible in a lower-dimensional representation.
## Installation

At the Julia REPL:

```julia
Pkg.clone("https://github.com/lucianolorenti/SpectralClustering.jl.git")
```

## Description

The library provides functions that allow:
* Build the affinity matrix. [Simmilarity graph creation](@ref), [Graph matrices](ref)
* Perform the embedding of the patterns in the space spanned by the eigenvectors of the matrices derived from the affinity matrix. [Eigenvector Embedding](@ref)
    * Obtain an approximation of the eigenvector in order to reduce the computational complexity. [Approximate embedding](@ref)
    * Exploiting information from multiple views. Corresponding nodes in each graph should have the same cluster membership. [MultiView embedding](@ref)
* Clusterize the eigenvector space. [Eigenvector Clustering](@ref)



# Bibliography
```@eval
import Documenter.Documents.RawHTML
using DocUtils
RawHTML(bibliography(["ng2002spectral","shi2000normalized","yu2001understanding"]))
```
