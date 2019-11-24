# Simmilarity graph creation
A weighted graph is an ordered pair $G=(V,E)$ that is composed of a set $V$ of vertices together with a set $E$ of edges $(i,j,w)$ $i,j \in V,w \in R$. The number $w$, the weight, represent the simmilarity between $i$ and $j$.

In order to build a simmilarity graph two elements have to be defined:

1. Which are the neighbors for a given vertex. For this, a concrete type that inherit from [`NeighborhoodConfig`](@ref SpectralClustering.VertexNeighborhood)  has to be instantiated. 

2. The simmilarity function between patterns.  The function receives the element being evaluated and its neighbors and returns a vector with the simmilarities between them.  The signature of the function has to be the following `function weight(i::Integer, j::Vector{Integer}, e1, e2)` where `i::Int` is the index of the pattern being evaluated, `j::Vector{Integer}`  are the indices of the neighbors of `i`;  `e1` are the `i`-th pattern and  `e2` are the  neighbors patterns.


# Examples
[Graph creation examples](../../../notebooks/Graph creation.html)




# Bibliography
```@eval
import Documenter.Documents.RawHTML
Base.include(@__MODULE__, "DocUtils.jl")
RawHTML(bibliography(["Zelnik-manor04self-tuningspectral"]))
```
# Reference
## Index
```@index
Modules=[SpectralClustering]
Pages   = ["man/graphcreation.md"]
```
## Content
```@autodocs
Modules=[SpectralClustering]
Pages=["Graph/Creation.jl"]
```
