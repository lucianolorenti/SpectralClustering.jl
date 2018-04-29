# Eigenvector Clustering

Once the eigenvectors are obtained, we have a continuous solution for a discrete problem. In order to obtain an assigment for every pattern,  it is needed to discretize the eigenvectors.
Obtaining this discrete solution from eigenvectors often requires solving another clustering problem, albeit in a lower-dimensional space. That is, eigenvectors are treated as geometrical coordinates of a point set.

This library provides two methods two obtain the discrete solution:
- Kmeans by means of [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
- The one proposed in [Multiclass spectral clustering](#stella2003multiclassv)

# Examples

[Eigenvector clusterization examples](../../notebooks/Eigenvector Clustering.html )

# Reference Index
```@index
Modules = [SpectralClustering]
Pages=["man/data_access.md"]
```

# Members Documentation
 
```@autodocs
Modules = [SpectralClustering]
Pages=["EigenvectorClustering.jl"]
```
# Bibliography
```@eval
import Documenter.Documents.RawHTML
using DocUtils
RawHTML(bibliography(["stella2003multiclass"]))
```
