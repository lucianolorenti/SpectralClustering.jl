using Clustering,
    LightGraphs,
    SparseArrays,
    LinearAlgebra,
    Arpack
export NystromMethod,
       LandmarkBasedRepresentation,
       DNCuts


"""
Large Scale Spectral Clustering with Landmark-Based Representation
Xinl ei Chen Deng Cai

# Members
- `landmark_selector::{T <: AbstractLandmarkSelection}` Method for extracting landmarks
- `number_of_landmarks::Integer` Number of landmarks to obtain
- `n_neighbors::Integer` Number of nearest neighbors
- `nev::Integer` Number of eigenvectors
- `w::Function` Number of clusters to obtain
- `normalize::Bool`
"""
struct LandmarkBasedRepresentation{T <: AbstractLandmarkSelection}
    landmark_selector::T
    number_of_landmarks::Integer
    n_neighbors::Integer
    nev::Integer
    w::Function
    normalize::Bool
end

"""
```
embedding(cfg::LandmarkBasedRepresentation, X)
```


"""
function embedding(cfg::LandmarkBasedRepresentation, X)

    n = number_of_patterns(X)
    p = cfg.number_of_landmarks
    landmark_indices = select_landmarks(cfg.landmark_selector, p, X)
    landmarks = get_element(X, landmark_indices)
    neighbors_cfg = KNNNeighborhood(landmarks, cfg.n_neighbors)


    I = zeros(Integer, n*cfg.n_neighbors)
    J = zeros(Integer, n*cfg.n_neighbors)
    V = zeros(n*cfg.n_neighbors)
    for i = 1:n
        i_neighbors =  neighbors(neighbors_cfg, get_element(X, i))
        weights = cfg.w(i, [], get_element(X, i), get_element(landmarks, i_neighbors))
        weights ./= sum(weights)
        for (j, (neigh, w)) in enumerate(zip(i_neighbors, weights))
            I[(i-1)*cfg.n_neighbors + j] = neigh
            J[(i-1)*cfg.n_neighbors + j] = i
            V[(i-1)*cfg.n_neighbors + j] = w
        end
    end
    Z = sparse(I, J, V, p, n)

    (svd) = svds(Z*Z', nsv=cfg.nev)[1]
    v = svd.S
    S = spdiagm(0=>1 ./ (sqrt.(v)))[1]
    B = S * svd.U' * Z
    if cfg.normalize
        B = normalize_cols(B)
    end
    return B'
end

"""
```julia

type NystromMethod{T<:AbstractLandmarkSelection}
landmarks_selector::T
number_of_landmarks::Integer
w::Function
nvec::Integer
end
```
The type ```NystromMethod``` proposed in  Spectral Grouping Using the Nystrom Method by Charless
Fowlkes, Serge Belongie, Fan Chung, and Jitendra Malik. It has to be defined:

- `landmarks_selector::T<:AbstractLandmarkSelection`. A mechanism to select the sampled
points.
- `number_of_landmarks::Integer`. The number of points to sample
- `w::Function`. The weight function for compute the similiarity. The signature of the weight function has to be `weight(i, j, e1,e2)`. Where `e1` and `e2` ara the data elements i-th and j-th respectivily, obtained via `get_element`, usually is a vector.
- `nvec::Integer`. The number of eigenvector to obtain.
- `threaded::Bool`. Default: True. Specifies whether the threaded version is used.
"""
struct NystromMethod{T <: AbstractLandmarkSelection} <: EigenvectorEmbedder
    landmarks_selector::T
    number_of_landmarks::Integer
    w::Function
    nvec::Integer
    threaded::Bool
end
function NystromMethod(landmarks_selector::T,
                       number_of_landmarks::Integer,
                       w::Function,
                       nvec::Integer) where T <: AbstractLandmarkSelection
    return NystromMethod(landmarks_selector,
                         number_of_landmarks,
                         w,
                         nvec,
                         true)
end

"""
create_A_B(cfg::NystromMethod, landmarks::Vector{Int},X)

Arguments:
- `cfg::NystromMethod`. The method configuration.
- `landmarks::Vector{T}`. A vector of integer that containts the \$n\$ indexes sampled from the data.
- `X` is the data that containt \$ N \$ patterns.

Let \$ W \\in \\mathbb{R}^{N \\times N}, W = \\begin{bmatrix} A & B^T \\\\ B & C \\end{bmatrix}, A
\\in \\mathbb{R}^{ n \\times n }, B \\in \\mathbb{R}^{(N-n) \\times n}, C \\in
\\mathbb{R}^{(N-n)\\times (N-n)} \$ . \$A\$ represents the subblock of weights among the random
samples, \$B\$ contains the weights from the random samples to the rest of the pixels, and
\$C\$ contains the weights between all of the remaining pixels.
The function computes \$A\$ and \$B\$ from the data ```X``` using the weight function defined in
```cfg```.
"""
function create_A_B(cfg::NystromMethod, landmarks::Vector{<:Integer}, X)
    n = number_of_patterns(X)
    p = length(landmarks)
    indexes_b = setdiff(collect(1:n), landmarks)
    m = length(indexes_b)
    n = p
    A = zeros(Float32, p, p)
    B = zeros(Float32, p, m)
    qq = length(get_element(X, 1))
    landmarks_m = zeros(Float32, length(get_element(X, 1)), length(landmarks))
    # Get a copy of the landamrks
    for j = 1:length(landmarks)
        get_element!(view(landmarks_m, :, j), X, landmarks[j])
    end
    Threads.@threads for j = 1:length(landmarks)
        A[:,j] = cfg.w(landmarks[j], landmarks, view(landmarks_m, :, j), landmarks_m)
    end

    vec_k = zeros(Float32, length(get_element(X, 1)), Threads.nthreads())
    Threads.@threads for k = 1:length(indexes_b)
       thread_id = Threads.threadid()
       get_element!(view(vec_k, :, thread_id), X, indexes_b[k])
       B[:,k] = cfg.w(indexes_b[k], landmarks, view(vec_k, :, thread_id), landmarks_m)
    end
    return (A, B)
end

function create_A_B_single_thread(cfg::NystromMethod, landmarks::Vector{<:Integer}, X)
    n = number_of_patterns(X)
    p = length(landmarks)
    indexes_b = setdiff(collect(1:n), landmarks)
    m  = length(indexes_b)
    n = p
    A = zeros(Float32, p, p)
    B = zeros(Float32, p, m)
    qq = length(get_element(X, 1))
    landmarks_m = zeros(Float32, length(get_element(X, 1)), length(landmarks))
    for j = 1:length(landmarks)
        get_element!(view(landmarks_m, :, j), X, landmarks[j])
    end
    for j = 1:length(landmarks)
        A[:,j] = cfg.w(landmarks[j], landmarks, view(landmarks_m, :, j), landmarks_m)
    end
    vec_k = zeros(Float32, length(get_element(X, 1)))
    for k = 1:length(indexes_b)
       get_element!(vec_k, X, indexes_b[k])
       B[:,k] = cfg.w(indexes_b[k], landmarks, vec_k, landmarks_m)
    end
    return (A, B)
end
"""
create_A_B(cfg::NystromMethod, X)

# Arguments:
- `cfg::NystromMethod`
- `X`

#Return values
- Sub-matrix A
- Sub-matrix B
- `Vector{Int}`. The sampled points used build the sub-matrices

This is an overloaded method. Computes the submatrix A and B according to
[`create_A_B(::NystromMethod, ::Vector{Int}, ::Any)`](@ref).
Returns the two submatrices and the sampled points used to calcluate it
"""
function create_A_B(cfg::NystromMethod, X)
  landmarks = select_landmarks(cfg.landmarks_selector, cfg.number_of_landmarks, X)
  if (cfg.threaded)
      (A, B) = create_A_B(cfg::NystromMethod, landmarks, X)
  else
      (A, B) = create_A_B_single_thread(cfg::NystromMethod, landmarks, X)
  end
  return (A, B, landmarks)
end
"""
embedding(cfg::NystromMethod, X)

This is an overloaded function
"""
function embedding(cfg::NystromMethod, X)
    n = number_of_patterns(X)
    landmarks = select_landmarks(cfg.landmarks_selector,
                                 cfg.number_of_landmarks,
                                 X)
    return embedding(cfg, landmarks, X)
end
"""
embedding(cfg::NystromMethod, landmarks::Vector{Int}, X)
# Arguments
- `cfg::[NystromMethod](@ref)`
- `landmarks::Vector{Int}`
- `x::Any`

# Return values
- `(E, L)`: The approximated eigenvectors, the aprooximated eigenvalues
Performs the eigenvector embedding according to
"""
function embedding(cfg::NystromMethod, landmarks::Vector{<:Integer}, X)
    if (cfg.threaded)
        (A, B) = create_A_B(cfg, landmarks, X)
    else
        (A, B) = create_A_B_single_thread(cfg, landmarks, X)
    end
    return embedding(cfg, A, B, landmarks)
end

function compute_dhat(AA::Matrix{T}, BB::Matrix{T}) where T
    n = size(AA, 1)
    m = size(BB, 2)
    dhat = zeros(T, n + m)
    dhat[1:n] = sum(AA, dims = 1) + sum(BB, dims = 2)'
    dhat[n + 1:end] = sum(BB, dims = 1) + sum(BB, dims = 2)' * pinv(AA) * BB
    dhat[dhat .< 0] .= 0
    return 1 ./ (sqrt.(dhat) .+ eps())
end
function compute_V(AA::Matrix{T}, BB::Matrix{T}, nvec::Integer) where T <: Number
    n = size(AA, 1)
    m = size(BB, 2)
    Asi = real(sqrt(Symmetric(pinv(AA))))
    F = svd(AA + ((Asi * (BB * BB')) * Asi))
    V_1 = (Asi * F.U) .* vec((1 ./ (sqrt.(F.S) .+ eps())))'
    VA = AA * V_1[:,1:nvec + 1]
    VB = BB' * V_1[:,1:nvec + 1]
    return vcat(VA, VB)
end
function normalize_A_and_B!(AA::Matrix, BB::Matrix)
    n    = size(AA, 1)
    m    = size(BB, 2)
    dhat = compute_dhat(AA, BB)
    vv   = view(dhat, 1:n)
    vb   = view(dhat, n .+ (1:m))
    for I in CartesianIndices(size(AA))
        @inbounds AA[I] *= vv[I[1]] * vv[I[2]]
    end
    for I in CartesianIndices(size(BB))
        @inbounds BB[I] *= vv[I[1]] * vb[I[2]]
    end
end
"""
embedding(cfg::NystromMethod, A::Matrix, B::Matrix, landmarks::Vector{Int})

Performs the eigenvector approximation given the two submatrices A and B.
"""
function embedding(cfg::NystromMethod, AA::Matrix, BB::Matrix, landmarks::Vector{<:Integer})
    n = size(AA, 1)
    m = size(BB, 2)
    normalize_A_and_B!(AA, BB)
    V = compute_V(AA, BB, cfg.nvec)
    indexes_b = setdiff(collect(1:(n + m)), landmarks)
    indexes   = sortperm(vcat(landmarks, indexes_b))
    for i = 2:cfg.nvec + 1
        V[:,i] = V[:,i] ./ V[:,1]
    end
    return V[indexes,2:cfg.nvec + 1]

end
"""
```@julia
type DNCuts
```
# Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation
## Jordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Member, Ferran Marques, Jitendra Malik
"""
struct DNCuts <: EigenvectorEmbedder
    scales::Integer
    nev::Integer
    img_size
end
function pixel_decimate(img_size::Tuple{Int,Int}, steps)
    (nr, nc) = img_size
    decimated = CartesianIndices(img_size)[1:steps:nr, 1:steps:nc]
    return (LinearIndices(img_size)[decimated][:], size(decimated))
end

embedding(d::DNCuts, g::Graph) = embedding(d, adjacency_matrix(g))

"""
```
embedding(d::DNCuts, L)
```
"""
function embedding(d::DNCuts, W::AbstractMatrix)
    matrices = []
    img_size = d.img_size
    for j = 1:d.scales
        (idx, img_size) = pixel_decimate(img_size,  2)
        B = W[:, idx]
        C = Diagonal(vec(1 ./ sum(B, dims=2))) * B
        push!(matrices, C)
        W = C' * B
    end
    ss = ShiMalikLaplacian(d.nev)
    V  = Arpack.eigs(ss, NormalizedLaplacian(NormalizedAdjacency(CombinatorialAdjacency(W))))
    for s = d.scales:-1:1
        V = matrices[s] * V
    end
    return svd_whiten(V)
end
