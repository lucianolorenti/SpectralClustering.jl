export embedding,
       NgLaplacian,
       ShiMalikLaplacian,
       YuShiPopout,
       PartialGroupingConstraints

abstract type AbstractEmbedding <: EigenvectorEmbedder
end

doc"""
```julia
type NgLaplacian <: AbstractEmbedding
```
# Members
- `nev::Integer`. The number of eigenvectors to obtain

Given a affinity matrix \$ W \\in \\mathbb{R}^{n \\times n} \$.  Ng et al defines the laplacian as \$ L =  D^{-\\frac{1}{2}} W D^{-\\frac{1}{2}} \$ where \$ D \$ is a diagonal matrix whose (i,i)-element is the sum of W's i-th row.

The embedding function solves a relaxed version of the following optimization problem:
$\begin{array}{crclcl}
    \displaystyle \max_{ U \in \mathbb{R}^{n\times k} \hspace{10pt} } & \mathrm{Tr}(U^T L  U)  &\\
   \textrm{s.a.}  {U^T U}  =   I &&
\end{array}$

U is a matrix that contains the `nev`  largest eigevectors of \$ L \$.

# References
- [On Spectral Clustering: Analysis and an algorithm. Andrew Y. Ng, Michael I. Jordan, Yair Weiss](http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf)
"""
type NgLaplacian <: AbstractEmbedding
    nev::Integer
end
"""
```julia
embedding(cfg::NgLaplacian, gr::Graph)
```
Performs the eigendecomposition of the matrix \$ L \$ derived from the graph `gr`. The matrix \$ L \$ is defined according to [`NgLaplacian`](@ref)

"""
function embedding(cfg::NgLaplacian, gr::Graph)
    (L, a) = ng_laplacian(gr)
    a=nothing
    return embedding(cfg,L)
end
"""
```julia
embedding(cfg::NgLaplacian, L::Union{Matrix,SparseMatrixCSC})
```
Performs the eigendecomposition of the matrix \$ L \$ defined according to [`NgLaplacian`](@ref)
"""
function embedding(cfg::NgLaplacian, L::Union{Matrix,SparseMatrixCSC})
    (vals,vec) = eigs(L,nev  = cfg.nev, which = :LM, maxiter=1000)
    vec        = real(vec)
    if cfg.nev == 1
      return vec
    else
      return vec./ mapslices(norm,vec,2)
    end
end
doc"""
The normalized laplacian as defined in  \$ D^{-\\frac{1}{2}} (D-W) D^{-\\frac{1}{2}} \$.

## References:
- Spectral Graph Theory. Fan Chung
- Normalized Cuts and Image Segmentation. Jiambo Shi and Jitendra Malik

```
type ShiMalikLaplacian <: AbstractEmbedding
```

# Members
- `nev::Integer`. The number of eigenvector to obtain.

"""
type ShiMalikLaplacian <: AbstractEmbedding
    nev::Integer
end

"""
```
struct PartialGroupingConstraints <: AbstractEmbedding
```

# Members

- `nev::Integer`. The number of eigenvector to obtain.

Segmentation Given Partial Grouping Constraints
Stella X. Yu and Jianbo Shi
"""
struct PartialGroupingConstraints <: AbstractEmbedding
    nev::Integer
end

"""
```
struct PGCMatrix{T,I,F} <: AbstractMatrix{T}
```

Partial grouping constraint structure. This sturct is passed to eigs to
performe the L*x computation according to (41), (42) and (43) of
""Segmentation Given Partial Grouping Constraints""
"""
struct PGCMatrix{T,I,F} <: AbstractMatrix{T}
    W::SparseMatrixCSC{T, I}
    At::Matrix{F}
end
import Base.size
import Base.issymmetric
import Base.*
import Base.A_mul_B!
function size(a::PGCMatrix)
    return size(a.W)
end
function issymmetric(a::PGCMatrix)
    return true
end
function A_mul_B!(dest::AbstractVector, a::PGCMatrix, x::AbstractVector)
    local z = x - a.At'*(a.At*x)
    local y = a.W*z
    dest[:]= y - a.At'*(a.At*y)
end

function restriction_matrix(nv::Integer,  restrictions::Vector{Vector{Integer}})

    local number_of_restrictions = length(restrictions)
    local U = spzeros(nv,nv)
    local k = 0
    for j=1:number_of_restrictions
        local U_t = restrictions[j]
        for s=1:length(U_t)-1
            k             = k+1
            U[U_t[s],k]   = 1
            U[U_t[s+1],k] = 1
        end
    end
    return U
end

function embedding(cfg::PartialGroupingConstraints, L::SparseMatrixCSC,Dsqrtinv::SparseMatrixCSC, restrictions::Vector{Vector{Integer}})
    U                 = restriction_matrix(size(L,1), restrictions)
    (svd, n)          = svds(Dsqrtinv*U, nsv = length(restrictions))
    (eigvals, eigvec) = eigs(PGCMatrix(L,svd.Vt),nev = cfg.nev, maxiter = 50000, which = :LM)
    eigvec            = real(eigvec)
    return Dsqrtinv*eigvec
end
function embedding(cfg::PartialGroupingConstraints, gr::Graph, restrictions::Vector{Vector{Integer}})
    (L, dinv)  = ng_laplacian(gr)
    return embedding(cfg, L, dinv, restrictions)
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
    U                 = restriction_matrix(number_of_vertices(grA), restrictions)
    (Wa,da)           = weight_matrix(Float32,grA)
    (Wr,dr)           = weight_matrix(Float32,grR)
    local da          = vec(sum(Wa,1))
    local dr          = vec(sum(Wr,1))
    local W           = Wa-Wr + spdiagm(dr)
    local Dsqrtinv    = spdiagm(1./sqrt.(da+dr))
    W                 = Dsqrtinv*W*Dsqrtinv
    return embedding(PartialGroupingConstraints(cfg.nev), W, Dsqrtinv, restrictions)
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
    local Wa       = nothing
    local Wr       = nothing
    (Wa,da)        = weight_matrix(Float32,grA)
    (Wr,dr)        = weight_matrix(Float32,grR)
    local Weq      = Wa-Wr + spdiagm(dr)
    local Deq      = spdiagm(da+dr)
    Wa             = nothing
    da             = nothing
    Wr             = nothing
    dr             = nothing
    (eigvals, vec) = eigs(Weq, Deq, nev = cfg.nev, tol=0.000001, maxiter = 10000, which = :LM)
    indexes        = sortperm(real(eigvals))
    vec            = real(vec[:,indexes])
    return vec./ mapslices(norm,vec,2)
end
"""
```julia
embedding(cfg::ShiMalikLaplacian, gr::Union{Graph,SparseMatrixCSC})
```
# Parameters
-   `cfg::ShiMalikLaplacian`. An instance of a [`ShiMalikLaplacian`](@ref)  that specify the number of eigenvectors to obtain
- `gr::Union{Graph,SparseMatrixCSC}`. The `Graph`(@ref Graph) or the weight matrix of wich is going to be computed the normalized laplacian matrix.

Performs the eigendecomposition of the normalized laplacian matrix of
the graph `gr` defined acoording to [`ShiMalikLaplacian`](@ref). Returns
the cfg.nev eigenvectors associated with the non-zero smallest
eigenvalues.
"""
function embedding(cfg::ShiMalikLaplacian, gr::Union{Graph,SparseMatrixCSC})
    (L, Dinvhalf)  = normalized_laplacian(gr)
    (vals,vec) = eigs(L,nev  = cfg.nev + 50, which = :SM, maxiter=1000)
    idxs = find(real(vals).>0.0000001)
    idxs = idxs[1:min(length(idxs),cfg.nev)]
    vec=Dinvhalf*real(vec[:,idxs])
    return vec./ mapslices(norm,vec,2)
end
"""
```
embedding{T<:AbstractEmbedding}(cfg::T, neighborhood::VertexNeighborhood, oracle::Function, data)
```
"""
function embedding{T<:AbstractEmbedding}(cfg::T, neighborhood::VertexNeighborhood,  oracle::Function, data)
    graph = create(cfg.graph_creator,data)
    return embedding(cfg,graph)
end

