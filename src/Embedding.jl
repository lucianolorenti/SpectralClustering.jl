export embedding,
       NgLaplacian,
       ShiMalikLaplacian,
       YuShiPopout,
       PartialGroupingConstraints

using LightGraphs
using LightGraphs.LinAlg
using ArnoldiMethod
using LinearAlgebra
using Arpack
abstract type AbstractEmbedding <: EigenvectorEmbedder
end

"""
```julia
type NgLaplacian <: AbstractEmbedding
```
# Members
- `nev::Integer`. The number of eigenvectors to obtain
- `normalize::Bool`. Wether normalize the obtained eigenvectors

Given a affinity matrix `` W \\in \\mathbb{R}^{n \\times n} ``.  Ng et al defines the laplacian as `` L =  D^{-\\frac{1}{2}} W D^{-\\frac{1}{2}} `` where `` D `` is a diagonal matrix whose (i,i)-element is the sum of W's i-th row.

The embedding function solves a relaxed version of the following optimization problem:
``\\begin{array}{crclcl}
    \\displaystyle \\max_{ U \\in \\mathbb{R}^{n\\times k} \\hspace{10pt} } & \\mathrm{Tr}(U^T L  U)  &\\\\
   \\textrm{s.a.}  {U^T U}  =   I &&
\\end{array}``

U is a matrix that contains the `nev`  largest eigevectors of `` L ``.

# References
- [On Spectral Clustering: Analysis and an algorithm. Andrew Y. Ng, Michael I. Jordan, Yair Weiss](http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf)
"""
struct NgLaplacian <: AbstractEmbedding
    nev::Integer
    normalize::Bool
end
NgLaplacian(nev::Integer) = NgLaplacian(nev, true)


"""
```julia
embedding(cfg::NgLaplacian, W::CombinatorialAdjacency)
```
Performs the eigendecomposition of the laplacian matrix of the weight matrix `` W `` defined according to [`NgLaplacian`](@ref)
"""
function embedding(cfg::NgLaplacian, W::CombinatorialAdjacency)
    return embedding(cfg, NormalizedAdjacency(W))
end

"""
```julia
embedding(cfg::NgLaplacian, L::NormalizedAdjacency)
```
Performs the eigendecomposition of the laplacian matrix of the weight matrix `` W `` defined according to [`NgLaplacian`](@ref)
"""
embedding(cfg::NgLaplacian, L::NormalizedAdjacency) = embedding(cfg, sparse(L))

"""
```julia
embedding(cfg::NgLaplacian, L::SparseMatrixCSC)
```
Performs the eigendecomposition of the laplacian matrix of the weight matrix `` W `` defined according to [`NgLaplacian`](@ref)
"""
function embedding(cfg::NgLaplacian, L::AbstractMatrix)
    (vals, vec) = LightGraphs.eigs(L, nev = cfg.nev + 5, which = LM(), restarts=5000)
    vec = real(vec)
    a = (.!(isapprox.(vals, 1)))
    vec  = vec[:, a]
    vec = vec[:, 1:cfg.nev]
    if cfg.normalize
        return normalize_rows(vec)
    else
        return vec
    end
end

"""
The normalized laplacian as defined in  `` D^{-\\frac{1}{2}} (D-W) D^{-\\frac{1}{2}} ``.

## References:
- Spectral Graph Theory. Fan Chung
- Normalized Cuts and Image Segmentation. Jiambo Shi and Jitendra Malik

```
type ShiMalikLaplacian <: AbstractEmbedding
```

# Members
- `nev::Integer`. The number of eigenvector to obtain.
- `normalize::Bool`. Wether normalize the obtained eigenvectors

"""
struct ShiMalikLaplacian <: AbstractEmbedding
    nev::Integer
    normalize::Bool
end
ShiMalikLaplacian(nev::Integer) = ShiMalikLaplacian(nev, true)

"""
```
struct PartialGroupingConstraints <: AbstractEmbedding
```

# Members

- `nev::Integer`. The number of eigenvector to obtain.
- `smooth::Bool`. Whether to user Smooth Constraints
- `normalize::Bool`. Whether to normalize the rows of the obtained vectors

Segmentation Given Partial Grouping Constraints
Stella X. Yu and Jianbo Shi
"""
struct PartialGroupingConstraints <: AbstractEmbedding
    nev::Integer
    smooth::Bool
    normalize::Bool
end

function PartialGroupingConstraints(nev::Integer; smooth::Bool=true, normalize::Bool=false)
    return PartialGroupingConstraints(nev, smooth, normalize)
end

"""
```
struct PGCMatrix{T,I,F} <: AbstractMatrix{T}
```

Partial grouping constraint structure. This sturct is passed to eigs to
performe the L*x computation according to (41), (42) and (43) of
""Segmentation Given Partial Grouping Constraints""
"""
struct PGCMatrix{T} <: AbstractMatrix{T}
    W::NormalizedAdjacency{T}
    At
end
import Base.size
import LinearAlgebra.issymmetric
import Base.*
import LinearAlgebra.mul!
function size(a::PGCMatrix)
    return size(a.W)
end
function issymmetric(a::PGCMatrix)
    return true
end
function mul!(dest::AbstractVector, a::PGCMatrix, x::AbstractVector)
    z = x - (a.At * x)
    y = a.W * z
    dest[:] = y - (a.At* y)
end

function restriction_matrix(nv::Integer,  restrictions::Vector{Vector{Integer}})
    I = Integer[]
    J = Integer[]
    V = Float64[]
    k = 0
    for j = 1:length(restrictions)
        U_t = restrictions[j]
        for i=1:length(U_t)-1
            elem_i = U_t[i]
            elem_j = U_t[i+1]
            k += 1
            push!(I, elem_i)
            push!(J, k)
            push!(V, 1.0)

            push!(I, elem_j)
            push!(J, k)
            push!(V, -1.0)
        end
    end
    return sparse(I, J, V, nv, k)
end
"""
```
function embedding(cfg::PartialGroupingConstraints, L::NormalizedAdjacency, restrictions::Vector{Vector{Integer}})
```

# Arguments
- `cfg::PartialGroupingConstraints`
- `L::NormalizedAdjacency`
- `restrictions::Vector{Vector{Integer}}`
"""
function embedding(cfg::PartialGroupingConstraints, L::NormalizedAdjacency, restrictions::Vector{Vector{Integer}})
    U = restriction_matrix(size(L, 1), restrictions) 
    if (cfg.smooth)
        U = sparse(L.A)' * U
    end
    DU = sparse(spdiagm(0=>prescalefactor(L)) * U)
    F = svds(DU, nsv=size(U, 2)-1)[1]
    AAt = sparse(F.U)*sparse(F.U)'
    (eigvals, eigvec) = LightGraphs.eigs(PGCMatrix(L, AAt), nev = cfg.nev, which = LM())
    eigvec = real(eigvec)
    V = spdiagm(0=>prescalefactor(L)) * eigvec
    println(size(V))
    if cfg.normalize
        return normalize_rows(V)
    else
        return V
    end
end
"""
```
embedding(cfg::PartialGroupingConstraints, gr::Graph, restrictions::Vector{Vector{Integer}})
```

# Arguments
- `cfg::PartialGroupingConstraints`
- `L::NormalizedAdjacency`
-  `restrictions::Vector{Vector{Integer}}`
"""
function embedding(cfg::PartialGroupingConstraints, gr::Graph, restrictions::Vector{Vector{Integer}})
    L = NormalizedAdjacency(CombinatorialAdjacency(adjacency_matrix(gr)))
    return embedding(cfg, L, restrictions)
end

"""
```
struct YuShiPopout <: AbstractEmbedding
```

# Members

- `nev::Integer`. The number of eigenvector to obtain.

Understanding Popout through Repulsion
Stella X. Yu and Jianbo Shi
"""

struct YuShiPopout <: AbstractEmbedding
    nev::Integer
end
"""

```julia
function embedding(cfg::YuShiPopout,  grA::Graph, grR::Graph)
```

# References
- Grouping with Directed Relationships. Stella X. Yu and Jianbo Shi
- Understanding Popout through Repulsion. Stella X. Yu and Jianbo Shi
"""
function embedding(cfg::YuShiPopout, grA::Graph, grR::Graph, restrictions::Vector{Vector{Integer}})
    Wa = adjacency_matrix(grA)
    Wr = adjacency_matrix(grR)
    dr = vec(sum(Wr, 1))
    W = Wa - Wr + spdiagm(dr)
    return embedding(PartialGroupingConstraints(cfg.nev), NormalizedAdjacency(CombinatorialAdjacency(W)), restrictions)
end

"""
```julia
function embedding(cfg::YuShiPopout,  grA::Graph, grR::Graph)
```

# References
- Grouping with Directed Relationships. Stella X. Yu and Jianbo Shi
- Understanding Popout through Repulsion. Stella X. Yu and Jianbo Shi
"""
function embedding(cfg::YuShiPopout, grA::Graph, grR::Graph)
    Wa = adjacency_matrix(grA)
    Wr = adjacency_matrix(grR)
    dr = vec(sum(Wr, 1))
    da = vec(sum(Wa, 1))
    Weq = Wa - Wr + spdiagm(dr)
    Deq = spdiagm(da + dr)
    Wa = nothing
    da = nothing
    Wr = nothing
    dr = nothing
    (eigvals, eigvec) = LightGraphs.eigs(Weq, Deq, nev = cfg.nev, tol = 0.000001,  which = LM())
    indexes = sortperm(real(eigvals))
    eigvec = real(eigvec[:,indexes])
    return eigvec ./ mapslices(LinearAlgebra.norm, eigvec, dims = [2])
end



"""
```julia
embedding(cfg::NgLaplacian, W::CombinatorialAdjacency)
```
Performs the eigendecomposition of the laplacian matrix of the weight matrix `` W `` defined according to [`NgLaplacian`](@ref)
"""
function embedding(cfg::ShiMalikLaplacian, W::CombinatorialAdjacency)
    return embedding(cfg, NormalizedLaplacian(NormalizedAdjacency(W)))
end



"""
```julia
embedding(cfg::ShiMalikLaplacian, L::NormalizedLaplacian)
```
# Parameters
-   `cfg::ShiMalikLaplacian`. An instance of a [`ShiMalikLaplacian`](@ref)  that specify the number of eigenvectors to obtain
- `gr::Union{Graph,SparseMatrixCSC}`. The `Graph`(@ref Graph) or the weight matrix of wich is going to be computed the normalized laplacian matrix.

Performs the eigendecomposition of the normalized laplacian matrix of
the laplacian matriz `L` defined acoording to [`ShiMalikLaplacian`](@ref). Returns
the cfg.nev eigenvectors associated with the non-zero smallest
eigenvalues.
"""
function embedding(cfg::ShiMalikLaplacian, L::NormalizedLaplacian)
    (vals, V) = LightGraphs.eigs(sparse(L), nev=min(cfg.nev + 10, size(L, 1)), which = SR(), restarts=5000)
    idxs = findall(real(vals) .> 0.0000001)
    idxs = idxs[1:min(length(idxs), cfg.nev)]
    V = spdiagm(0 => L.A.A.D.^(1 / 2)) * real(V[:,idxs])
    if cfg.normalize
        return normalize_rows(V)
    else
        return V
    end
end

"""
```
embedding{T<:AbstractEmbedding}(cfg::T, neighborhood::VertexNeighborhood, oracle::Function, data)
```
"""
function embedding(cfg::T, neighborhood::VertexNeighborhood,  oracle::Function, data) where T <: AbstractEmbedding
    graph = create(cfg.graph_creator, data)
    return embedding(cfg, graph)
end


"""
```julia
embedding(cfg::T, gr::Graph) where T<:AbstractEmbedding
```
Performs the eigendecomposition of the laplacian matrix of the weight matrix `` W `` derived from the graph `gr` defined according to [`NgLaplacian`](@ref)
"""
function embedding(cfg::T, gr::Graph) where T <: AbstractEmbedding
    return embedding(cfg, CombinatorialAdjacency(adjacency_matrix(gr, dir = :both)))
end
