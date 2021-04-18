# SpectralClustering

- [Documentation](https://lucianolorenti.github.io/SpectralClustering.jl/latest)
- Check out the [Examples](https://lucianolorenti.github.io/SpectralClustering.jl/latest/notebooks/Index.html)
[![Coverage Status](https://coveralls.io/repos/github/lucianolorenti/SpectralClustering.jl/badge.svg?branch=master)](https://coveralls.io/github/lucianolorenti/SpectralClustering.jl?branch=master)

The library provides functions that allow:
* Build the affinity matrix.
* Perform the embedding of the patterns in the space spanned by the eigenvectors of the matrices derived from the affinity matrix.
    * Obtain an approximation of the eigenvectors in order to reduce the computational complexity.
    * Exploiting information from multiple views. Corresponding nodes in each graph should have the same cluster membership.
* Clusterize the eigenvector space.

# Methods implemented

* Graph construction
  * [Self-Tuning Spectral Clustering](https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf)
* Embedding
  * [Normalized cuts and image segmentation](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf)
  * [On Spectral Clustering: Analysis and an algorithm](https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf)
  * [Understanding Popout through Repulsion](https://pdfs.semanticscholar.org/019c/099ab01902416a625a9d18a36e61b88f5a3d.pdf)
  * [Segmentation Given Partial Grouping Constraints](http://www.cs.cmu.edu/~xingyu/papers/yu_bias.pdf)
* Approximate embedding
  * [Spectral grouping using the nystrom method](https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf)
     * [Nystrom sampling depends on the eigenspectrum shape of the data](https://openreview.net/pdf?id=HJZvjvJPf)
  * [Large Scale Spectral Clustering
with Landmark-Based Representation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.6933&rep=rep1&type=pdf)
* Multiple views
  * Kernel Addition
  * Kernel Product
  * Feature Concatenation (in the examples section)
  * [Co-regularized Multi-view Spectral Clustering](https://papers.nips.cc/paper/4360-co-regularized-multi-view-spectral-clustering.pdf)
* Incremental
  * TODO [Incremental spectral clustering by efficiently updating the eigen-system](https://www.sciencedirect.com/science/article/pii/S0031320309002209/pdfft?md5=dc50ecba5ab9ab23ea239ef89244800a&pid=1-s2.0-S0031320309002209-main.pdf)
* Clusterize
  * [Multiclass Spectral Clustering](https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf)
  * KMeans via [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)

The documentation and the library is still a work in progress.
