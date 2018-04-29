export clusterize,
       KMeansClusterizer,
       YuEigenvectorRotation,
       EigenvectorClusterizer,
       EigenvectorsClusteringResult

using Clustering


import Clustering.assignments
import Clustering.ClusteringResult
type EigenvectorClusteringResult{T<:Integer} <: ClusteringResult
    assignments::Vector{T}
end
function assignments(r::EigenvectorClusteringResult)
    return r.assignments
end
abstract type EigenvectorClusterizer end
"""
```julia
struct KMeansClusterizer <: EigenvectorClusterizer
    k::Integer
    init::Symbol
end
```
"""
struct KMeansClusterizer <: EigenvectorClusterizer
    k::Integer
    init::Symbol
end
function KMeansClusterizer(k::Integer)
    return KMeansClusterizer(k, :kmpp)
end
function clusterize(t::KMeansClusterizer, E)
    model = kmeans(E', t.k, init =t.init)

    return EigenvectorClusteringResult(assignments(model))
end

"""
Multiclass Spectral Clustering
"""
struct YuEigenvectorRotation <: EigenvectorClusterizer
    maxIter::Integer
end
function YuEigenvectorRotation()
    return YuEigenvectorRotation(500)
end
function clusterize(cfg::YuEigenvectorRotation,V::Matrix)
    (N,k)              = size(V)
    V                  = spdiagm(1./sqrt.(vec(mapslices(norm,V,2))))*V
    local hasConverged = false
    local R            = zeros(k,k)
    R[:,1]             = [ V[rand(1:N),i] for i = 1:k ]
    local c            = zeros(N)
    for j=2:k
        c = c+abs.(V*R[:,j-1])
        R[:, j] = V[findmin(c)[2], :]'
    end
    local lastObjectiveValue = Inf
    local nIter              = 0
    local ncut_value         = 0
    while !hasConverged
        nIter            = nIter+ 1
        t_discrete       = V*R
        #non maximum supression
        labels           = ind2sub(size(V),vec(findmax(V,2)[2]))[2]
        vectors_discrete = sparse(collect(1:N), labels,ones(Int64,length(labels)),      N,k)
        t_svd            = full(vectors_discrete' * vectors_discrete)
        U, S, Vh         = svd(t_svd)
        ncutValue        = 2.0 * (N - sum(S,1))
        if ((abs(ncutValue[1] - lastObjectiveValue) < eps()) || (nIter > cfg.maxIter))
            hasConverged = true
        else
            lastObjectiveValue = ncutValue[1]
            R= Vh'*U'
        end
    end
    labels = ind2sub(size(V),vec(findmax(V,2)[2]))[2]
    return EigenvectorClusteringResult(labels)
end
"""
```julia
function clusterize{T<:EigenvectorEmbedder, C<:EigenvectorClusterizer}(cfg::T, clus::C, X)
```

Given a set of patterns `X` generates an eigenvector space according to `T<:EigenvectorEmbeddder` and then clusterize the eigenvectors using the algorithm defined
by `C<:EigenvectorClusterize`.

"""
function clusterize(cfg::T, clus::C, X) where T<:EigenvectorEmbedder where C<:EigenvectorClusterizer
  local E = embedding(cfg,X)
  return clusterize(clus, E)
end
