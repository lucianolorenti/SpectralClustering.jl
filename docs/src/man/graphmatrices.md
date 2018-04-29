# Graph matrices
The `Graphs.Matrices`  module provides a collecion of functions that computes differents matrices  from a graph.
Given a `Graph` $G=(V,E)$ where $(v_i,v_j,w_{ij}) \in E, v_i,v_j \in V, w_{ij} \in R$ this module allows to obtain:
- The simmiliarty matrix $W$. Where $W_{ij} = w_{ij}$
- The laplacian matrix $L=(D-W)$. Where $D = W \mathbf{1}$
- The normalized laplacian $L_N = D^{-\frac{1}{2}} (D-W) D^{-\frac{1}{2}}$
- NSJ Laplacian  $L_{NSJ} = D^{-1}W$

## Detailed Description
### Laplacian
```julia

```
### Normalized Laplacian
### NSJ Laplacian

# Reference
## Index
```@contents
Pages=["Graph/Matrices.jl"]
Modules=[SpectralClustering]
```
## Content
```@autodocs

Pages=["Graph/Matrices.jl"]
Modules=[SpectralClustering]
```
