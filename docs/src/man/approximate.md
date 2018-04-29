# Approximate embedding
Given a symmetric affinity matrix \$A\$, we would like to compute the \$k\$ smallest eigenvectors of the Laplacian of A. Directly computing such eigenvectors can be very costly even with sophisticated solvers, due to the large size of \$A\$.

# Examples

[Approximate embedding examples](../../notebooks/Approximate Embedding.html )
# Bibliography
```@eval
import Documenter.Documents.RawHTML
using DocUtils
RawHTML(bibliography(["pont2017multiscale"]))
```
# Reference
## Index
```@index
Modules=[SpectralClustering]
Pages=["man/approximate.md"]
```

## Content
```@autodocs

Pages=["ApproximateEmbedding.jl"]
Modules=[SpectralClustering]
```
