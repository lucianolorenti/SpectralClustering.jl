export clusterize,
       KMeansClusterizer,
       YuEigenvectorRotation,
       EigenvectorClusterizer,
       EigenvectorClusteringResult

using Clustering


import Clustering.assignments
import Clustering.ClusteringResult
struct EigenvectorClusteringResult{T<:Integer} <: ClusteringResult
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
    model = kmeans(Matrix(E'), t.k, init =t.init)

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
function clusterize(cfg::YuEigenvectorRotation, X_star_hat::Matrix)
     (N,k) = size(X_star_hat)
     X_star_hat = spdiagm(0=>1 ./ sqrt.(vec(mapslices(norm, X_star_hat, dims=[2]))))*X_star_hat
     hasConverged = false
     R_star = zeros(k,k)
     R_star[:, 1] = [X_star_hat[rand(1:N), i] for i = 1:k ]
     c = zeros(N)
     for j=2:k
        c = c + abs.(X_star_hat*R_star[:,j-1])
        i = findmin(c)[2]
        R_star[:, j] = X_star_hat[i, :]'
     end
     lastObjectiveValue = Inf
     nIter = 0
     ncut_value = 0
     X_star = nothing
     while !hasConverged
        nIter = nIter+ 1
        X_hat = X_star_hat*R_star  
        #non maximum supression
        labels = vec([I[2] for I in findmax(X_hat, dims=2)[2]])
        X_star = zeros(size(X_star_hat))
        for (i, l) = enumerate(labels)
            X_star[i, l] = l
        end
        F = svd(X_star' * X_star_hat, full=true)
        U, S, Vh = (F.U, F.S, F.Vt)
        ncutValue = sum(S)
        if ((abs(ncutValue - lastObjectiveValue) < eps()) || (nIter > cfg.maxIter))
            hasConverged = true
        else
            lastObjectiveValue = ncutValue
            R_star = Vh'*U'            
        end
    end
    labels = vec([I[2] for I in findmax(X_star, dims=2)[2]])
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
    E = embedding(cfg,X)
    return clusterize(clus, E)
end
