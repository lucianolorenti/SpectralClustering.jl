using Clustering,
   LightGraphs
export NystromMethod,
       LandmarkBasedRepresentation,
       DNCuts
"""
Large Scale Spectral Clustering with Landmark-Based Representation
Xinl ei Chen Deng Cai

"""
type LandmarkBasedRepresentation{T<:AbstractLandmarkSelection}
  landmark_selector::T
  number_of_landmarks::Integer
  r::Integer #nearest landamrks
  k::Integer
end
"""
clusterize(cfg::LandmarkBasedRepresentation,X)

"""
function clusterize(cfg::LandmarkBasedRepresentation,X)
  n         = number_of_patterns(X)
  p         = number_of_landmarks(cfg.landmarks, X)
  landmarks = select_landmarks(cfg.landmarks,X)
  tree      = RTree(length(get_element(X,1)[1:2]))
  for i = 1:length(landmarks)
    vec_i  = get_element(X,landmarks[i])
    add_point!(tree,i,vec_i[1:2])
  end
  I = zeros(Integer,cfg.r*n)
  J = zeros(Integer,cfg.r*n)
  Z = zeros(Float64,cfg.r*n)
  c=1
  params = [15,0.2]
  for i=1:n
    vec_i = get_element(X,i)
    nn     = k_nearest_neighbors_id(tree,cfg.r,vec_i[1:2])
    for j in nn
      deno = sum( [w(vec_i, get_element(X,l),params) for l in nn])
      nume = cfg.w(vec_i, get_element(X,j),params)
      I[c] = j
      J[c] = i
      Z[c] = nume/(deno)
      c = c + 1
    end
  end
  Z=sparse(I,J,Z,p,n)
  p=zeros(p)
  (s,A) = eigs(Z*Z',nev=3,maxiter=1000, v0= v0)
  S=spdiagm(1./(sqrt(s)+eps()))
  B=S*A'*Z

  for j=1:size(B,2)
    B[:,j]=B[:,j]./(norm(B[:,j])+eps())
  end
  result = Clustering.kmeans(B,cfg.k, init=:kmcen)
  return result.assignments

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
type NystromMethod{T<:AbstractLandmarkSelection} <: EigenvectorEmbedder
  landmarks_selector::T
  number_of_landmarks::Integer
  w::Function
  nvec::Integer
  threaded::Bool
end
function NystromMethod{T<:AbstractLandmarkSelection}(
                                  landmarks_selector::T,
                                  number_of_landmarks::Integer,
                                  w::Function,
                                  nvec::Integer)
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
    n         = number_of_patterns(X)
    p         = length(landmarks)
    indexes_b = setdiff(collect(1:n), landmarks)
    m         = length(indexes_b)
    n         = p
    A         = zeros(Float32,p,p)
    B         = zeros(Float32,p,m)
    qq = length(get_element(X,1))
    landmarks_m = zeros(Float32,length(get_element(X,1)), length(landmarks))
    # Get a copy of the landamrks
    for j = 1:length(landmarks)
        get_element!(view(landmarks_m,:,j),X,landmarks[j])
    end
    Threads.@threads for j=1:length(landmarks)
        A[:,j] = cfg.w(landmarks[j], landmarks, view(landmarks_m,:,j), landmarks_m )
    end
    
    vec_k = zeros(Float32,length(get_element(X,1)), Threads.nthreads())
    Threads.@threads for k=1:length(indexes_b)
       thread_id = Threads.threadid()
       get_element!(view(vec_k,:,thread_id),X,indexes_b[k])
       B[:,k] = cfg.w(indexes_b[k], landmarks, view(vec_k,:,thread_id), landmarks_m)
    end
    return (A,B)
end

function create_A_B_single_thread(cfg::NystromMethod, landmarks::Vector{<:Integer}, X)
    local n         = number_of_patterns(X)
    local p         = length(landmarks)
    local indexes_b = setdiff(collect(1:n), landmarks)
    local m         = length(indexes_b)
    n               = p
    local A         = zeros(Float32,p,p)
    local B         = zeros(Float32,p,m)
    local qq = length(get_element(X,1))
    local landmarks_m = zeros(Float32,length(get_element(X,1)), length(landmarks))
    for j = 1:length(landmarks)
       get_element!(view(landmarks_m,:,j),X,landmarks[j])
    end
    for j=1:length(landmarks)
      A[:,j] = cfg.w(landmarks[j], landmarks, view(landmarks_m,:,j), landmarks_m )
    end
    local vec_k = zeros(Float32,length(get_element(X,1)))
    for k=1:length(indexes_b)
       get_element!(vec_k,X,indexes_b[k])
       B[:,k] = cfg.w(indexes_b[k], landmarks, vec_k, landmarks_m)
    end
    return (A,B)
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
      (A,B)     = create_A_B(cfg::NystromMethod,landmarks, X)
  else
      (A,B)     = create_A_B_single_thread(cfg::NystromMethod,landmarks, X)
  end
  return (A,B,landmarks)
end
"""
embedding(cfg::NystromMethod, X)

This is an overloaded function
"""
function embedding(cfg::NystromMethod, X)
  n         = number_of_patterns(X)
  landmarks = select_landmarks(cfg.landmarks_selector,cfg.number_of_landmarks,X)
  return embedding(cfg,landmarks,X)
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
    local A = nothing
    local B = nothing
    if (cfg.threaded)
        (A,B)     = create_A_B(cfg,landmarks,X)
    else
        (A,B) = create_A_B_single_thread(cfg, landmarks, X)
    end
    return embedding(cfg,A,B,landmarks)
end

function compute_dhat(AA::Matrix{T}, BB::Matrix{T}) where T
    local n    = size(AA,1)
    local m    = size(BB,2)
    local dhat = zeros(T,n+m)
    #d1        = sum(vcat(A,B'),1)
    dhat[1:n]  = sum(AA,1) + sum(BB,2)'
    #d2        = sum(B,1) + sum(B',1)*pinv(A)*B
    dhat[n+1:end] =  sum(BB,1) + sum(BB,2)'*pinv(AA)*BB
    #dhat= sqrt(1./(hcat(d1,d2)+eps()))'
    return 1./(sqrt.(dhat).+eps())
end
function compute_V(AA::Matrix{T}, BB::Matrix{T}, nvec::Integer) where T<:Number
    local n    = size(AA,1)
    local m    = size(BB,2)
    local Asi  = real(sqrtm(Symmetric(pinv(AA))))
    local F    = svdfact(  AA+((Asi*(BB*BB'))*Asi) )
    local V_1  = (Asi*F[:U]).*vec((1./(sqrt.(F[:S])+eps())))'
    local VA   = AA*V_1[:,1:nvec+1]
    local VB   = BB'*V_1[:,1:nvec+1]
    return vcat(VA,VB)
end
function normalize_A_and_B!(AA::Matrix, BB::Matrix)
    local n    = size(AA,1)
    local m    = size(BB,2)
    local dhat = compute_dhat(AA,BB)
    local vv   = view(dhat,1:n)
    local vb   = view(dhat,n+(1:m))
    for I in CartesianRange(size(AA))
        @inbounds AA[I] *=vv[I[1]]*vv[I[2]]
    end
    for I in CartesianRange(size(BB))
        @inbounds BB[I] *= vv[I[1]]*vb[I[2]]
    end
end
"""
embedding(cfg::NystromMethod, A::Matrix, B::Matrix, landmarks::Vector{Int})

Performs the eigenvector approximation given the two submatrices A and B.
"""
function embedding(cfg::NystromMethod, AA::Matrix, BB::Matrix, landmarks::Vector{<:Integer})
    local n         = size(AA,1)
    local m         = size(BB,2)
    normalize_A_and_B!(AA,BB)
    local V = compute_V(AA,BB, cfg.nvec)
    local indexes_b = setdiff(collect(1:(n+m)), landmarks)
    local indexes   = sortperm(vcat(landmarks,indexes_b))

    for i = 2:cfg.nvec+1
        V[:,i] = V[:,i] ./V[:,1]
    end
    return V[indexes,2:cfg.nvec+1]

end
"""
```@julia
type DNCuts
```
# Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation
## Jordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Member, Ferran Marques, Jitendra Malik
"""
type DNCuts
  scales::Integer
  nev::Integer
  img_size
end
function pixel_decimate(img_size::Tuple{Int,Int}, L, steps)
  (i,j) = ind2sub(img_size, 1:size(L,1))
  return find((mod.(i,steps) .== 0) .& (mod.(j,steps) .==0))
end

function svd_whiten(X)
    U, s, Vt = svd(X)
    return U*Vt
end
#=
# get the column sums of A
S = vec(sum(A,1))

# get the nonzero entries in A. ei is row index, ej is col index, ev is the value in A
ei,ej,ev = findnz(A)

# get the number or rows and columns in A
m,n = size(A)

# create a new normalized matrix. For each nonzero index (ei,ej), its new value will be
# the old value divided by the sum of that column, which can be obtained by S[ej]
A_normalized = sparse(ei,ej,ev./S[ej],m,n)
http://stackoverflow.com/questions/24296856/in-julia-how-can-i-column-normalize-a-sparse-matrix
=#
function normalize_cols(A)
	for (col	,s) in enumerate(sum(A,1))
          s == 0 && continue # What does a "normalized" column with a sum of zero look like?
         A[:,col] = A[:,col]/s
       end
       return A
end
function normalize_columns(A :: SparseMatrixCSC)
          sums = sum(A,1)+ eps()
          I,J,V = findnz(A)
          for idx in 1:length(V)
            V[idx] /= sums[J[idx]]
          end
          sparse(I,J,V)
end
"""
```
embedding(d::DNCuts, L)
```
"""
function embedding(d::DNCuts, W)
    local matrices = []
    local img_size = d.img_size
    for j=1:d.scales
        local idx = pixel_decimate(img_size,W,2)
        local B   = W[:,idx]
        local C   = normalize_columns(B')'
        push!(matrices,C)
        W = C'*B
        img_size = (round(Int,img_size[1]/2), round(Int,img_size[2]/2))
    end
    local ss = NgLaplacian(d.nev)
    W = (W+W')/2
    local V  = real(embedding(ss, CombinatorialAdjacency(W)))
    for s=d.scales:-1:1
        V = matrices[s]*V
    end
    return svd_whiten(V)
end
