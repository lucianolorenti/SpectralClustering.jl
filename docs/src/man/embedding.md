# Eigenvector Embedding
Spectral clustering techniques require the computation of the extreme eigenvectors of matrices derived from patterns similarity . The Laplacian matrix obtained from the data is generally used as the starting point for decomposition into autovectors. Given the symmetric matrix $ W (i, j) = w_{ij}, W \in R^{n \times n} $ that contains information about  similarity between the patterns, if $ D = W \mathbf {1} $, the unnormalized Laplacian matrix is defined as $ L = D-W $.

The matrix $ W $ can be seen as the incidence matrix of a weighted graph. The [Simmilarity graph creation] (@ref) utilities
implement functions that allow the construction of simmilarty graphs.

The Embedding utilities contain the functions for performing the embedding of the patterns in the space spanned by the $ k $ eigenvectors of a matrix derived from $W$.

Currently the module implements the techniques described in:

- [On spectral clustering: Analysis and an algorithm.](#ng2002spectral)
- [Normalized cuts and image segmentation.](#shi2000normalized)
- [Understanding Popout through Repulsion.](#yu2001understanding)
- [Segmentation Given Partial Grouping Constraints](#yu2004segmentation)

# Examples
[Embedding examples](../../../notebooks/Embedding.html)

# Bibliography
```@eval
import Documenter.Documents.RawHTML
Base.include(@__MODULE__, "DocUtils.jl")
RawHTML(bibliography(["ng2002spectral","shi2000normalized","yu2001understanding", "yu2004segmentation","lee2007trajectory"]))
```
## Index
```@index
Modules=[SpectralClustering]
Pages=["man/embedding.md"]
```
## Content
```@autodocs
Modules=[SpectralClustering]
Pages=["src/Embedding.jl"]
```
