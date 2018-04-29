using NearestNeighbors

using StatsBase
export VertexNeighborhood,
       KNNNeighborhood,
       create,
       PixelNeighborhood,
       local_scale,
       neighbors,
       RandomNeighborhood


import SpectralClustering: spatial_position
"""
```julia
type RandomKGraph
```
The type RandomKGraph defines the parameters needed to create a random k-graph.
Every vertex it is connected to `k` random neigbors.
# Members
- `number_of_vertices::Integer`. Defines the number of vertices of the graph.
- `k::Integer`. Defines the minimum number of  neighborhood of every vertex.
"""
type RandomKGraph
    number_of_vertices::Integer
    k::Integer
end
"""
```julia
create(cfg::RandomKGraph)
```
Construct a [`RandomKGraph`](@ref) such that every vertex is connected with other k random vertices.
"""
function create(cfg::RandomKGraph)
    g = Graph(cfg.number_of_vertices)
    for i=1:cfg.number_of_vertices
        cant = 0
        while cant < cfg.k
            selected = rand(1:cfg.number_of_vertices)
            while selected == i
                selected = rand(1:cfg.number_of_vertices)
            end
            connect!(g,i,selected,rand())
            cant=cant +1
        end
    end
    return g;
end
"""
```julia
abstract type VertexNeighborhood end
```
The abstract type VertexNeighborhood provides an interface to query for the
neighborhood of a given vertex. Every concrete type that inherit from
VertexNeighborhood must define the function
```julia
neighbors{T<:VertexNeighborhood}(cfg::T, j::Integer, data)
```
which returns the neighbors list of the vertex j for the given data.
"""
abstract type VertexNeighborhood end
"""
```julia
type PixelNeighborhood  <: VertexNeighborhood
```
`PixelNeighborhood` defines neighborhood for a given pixel based in its spatial location. Given a pixel located at (x,y), returns every pixel inside
\$(x+e,y), (x-e,y)\$ and \$(x,y+e)(x,y-e)\$.

# Members
- e:: Integer. Defines the radius of the neighborhood.

"""
type PixelNeighborhood  <: VertexNeighborhood
    e::Integer
end
"""
```julia
neighbors(cfg::PixelNeighborhood, j::Integer, img)
```

Returns the neighbors of the pixel j according to the specified in [`PixelNeighborhood`](@ref)
"""
function neighbors(cfg::PixelNeighborhood, j::Integer, img::Matrix{T}) where T<: Colorant
    pos = ind2sub(size(img),j)
    w_r = max(pos[1]-cfg.e,1):min(pos[1]+cfg.e, size(img,1))
    w_c = max(pos[2]-cfg.e,1):min(pos[2]+cfg.e, size(img,2))
    return vec(map(x->sub2ind(size(img),x[1],x[2]),CartesianRange((w_r,w_c))))
end

"""
```julia
type CliqueNeighborhood <: VertexNeighborhood
```
`CliqueNeighborhood` specifies that the neighborhood for a given vertex \$j\$ in a
graph of \$n\$ vertices are the remaining n-1 vertices.
"""
type CliqueNeighborhood <: VertexNeighborhood
end
"""
```julia
neighbors(config::CliqueNeighborhood, j::Integer, X)
```
Return every other vertex index different from \$j\$. See [`CliqueNeighborhood`](@ref)
"""
function neighbors(config::CliqueNeighborhood,j::Integer,X)
  return filter!(x->x!=j,collect(1:number_of_patterns(X)))
end
"""
```julia
type KNNNeighborhood <: VertexNeighborhood
    k::Integer
    tree::SpatialIndex.Index
end
```
`KNNNeighborhood` specifies that the neighborhood for a given vertex \$j\$ are the \$k\$ nearest neighborgs. It uses a tree to search the nearest patterns.
# Members
- `k::Integer`. The number of k nearest neighborgs to connect.
- `tree::KDTree`. Internal data structure.
- `f::Function`. Transformation function
"""
type KNNNeighborhood <: VertexNeighborhood
  k::Integer
  tree::KDTree
  t::Function    
end
"""
```julia
KNNNeighborhood(X::Matrix, k::Integer)
```
Create the [`KNNNeighborhood`](@ref) type by building a `k`-nn tre from de data `X`

Return the indexes of the `config.k` nearest neigbors of the data point `j` of the data `X`.
"""
function KNNNeighborhood(X, k::Integer, f::Function=x->x)
    local tree = KDTree(hcat([f(get_element(X,j)) for j=1:number_of_patterns(X)]...))
    return KNNNeighborhood(k, tree, f)
end
function neighbors(config::KNNNeighborhood,j::Integer,X)
    idxs, dists = knn(config.tree, config.t(get_element(X,j)), config.k+1, true)
    return idxs[2:config.k+1]
end
"""
```@julia
type RandomNeighborhood <: VertexNeighborhood
    k::Integer
end
```
For a given index `j`return `k` random vertices different from `j`
"""
type RandomNeighborhood <: VertexNeighborhood
    k::Integer
end
function neighbors(config::RandomNeighborhood, j::Integer, X)
    local samples = StatsBase.sample(1:number_of_patterns(X),config.k,replace=false)
    if (in(j, samples))
        filter!(e->e!=j,samples)
    end
    while (length(samples) < config.k)
        local s  =  StatsBase.sample(1:number_of_patterns(X), 1)[1]
        if (s!=j)
            push!(samples,s)
        end
    end
    return samples
end
"""
```julia
weight{T<:DataAccessor}(w::Function,d::T, i::Int,j::Int,X)
```
Invoke the weight function provided to compute the similarity between the pattern `i` and the pattern `j`.
"""
function weight(w::Function,i::Integer,j::Integer,X)
  x_i = get_element(X,i)
  x_j = get_element(X,j)
  return w(i,j,x_i,x_j)
end

"""
```julia
create(w_type::DataType, neighborhood::VertexNeighborhood, oracle::Function,X)
```
Given a [`VertexNeighborhood`](@ref), a simmilarity function `oracle`  construct a simmilarity graph of the patterns in `X`.
"""
function create(w_type::DataType, neighborhood::VertexNeighborhood, oracle::Function,X)
    local number_of_vertices = number_of_patterns(X)
    local g                  = Graph(number_of_vertices; weight_type= w_type)
    @Threads.threads  for j=1:number_of_vertices
        local neigh   = neighbors(neighborhood,j,X)
        local x_j     = get_element(X,j)
        local x_neigh = get_element(X,neigh)
        local weights = oracle(j,neigh,x_j,x_neigh)
        connect!(g, j,neigh,weights)
    end
    gc()
    return g
end
"""
```julia
create(neighborhood::VertexNeighborhood, oracle::Function,X)
```
Given a [`VertexNeighborhood`](@ref), a simmilarity function `oracle` construct a simmilarity graph of the patterns in `X`.
"""
function create(neighborhood::VertexNeighborhood, oracle::Function, X)
    create(Float64, neighborhood, oracle, X)
end
"""
```julia
local_scale(neighborhood::KNNNeighborhood, oracle::Function, X; k = 7)
```
Computes the local scale of each pattern according to [Self-Tuning Spectral Clustering](https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf).
Return a matrix containing for every pattern the local_scale.

\"The selection of the local scale \$ \\sigma \$ can be done by studying the local statistics of the neighborhoods surrounding points \$ i \$ and \$ j \$ .i \"
Zelnik-Manor and Perona use \$ \\sigma_i = d(s_i, s_K) \$ where \$s_K\$ is the \$ K \$ neighbor of point \$ s_i \$ .
They \"used a single value of \$K=7\$, which gave good results even for high-dimensional data \" .

"""
function local_scale(neighborhood::KNNNeighborhood,oracle::Function,X; k=7)

  number_of_vertices = number_of_patterns(X)
  temp               = oracle(get_element(X,1), get_element(X,2))
  scales             = zeros(size(temp,1), number_of_vertices)
  for j =1:number_of_vertices
      neigh = neighbors(neighborhood,j,X)
      neigh = neigh[end]
      scales[:,j] = oracle(get_element(X,j), get_element(X,neigh))
  end
  return scales
end
function local_scale(neighborhood:: PixelNeighborhood,oracle::Function, img)

  number_of_vertices = number_of_patterns(img)
  temp               = oracle(1, img, (1,2))
  scales             = zeros(size(temp,1),number_of_vertices)

  for j=1:number_of_vertices
      neigh = spatial_position(img,neighbors(neighborhood,j,img))
      scales[:,j] = oracle(j, img, neigh)
  end
  return scales
end

#="""
Given a graph (g) created from a X_prev \in R^{d x n}, updates de graph from
the matrix X \in R^{d x m}, m > n. Adding the correspondent vertices and connecting
them whith the existing ones.
"""
function update!(config::GraphCreationConfig,g::Graph,X)
  number_of_vertices = number_of_patterns(config.da,X)
  old_number_of_vertices = number_of_vertices(g)
  for j=old_number_of_vertices+1:number_of_vertices
    add_vertex!(g)
  end
  for j=old_number_of_vertices+1:number_of_vertices
      neigh = neighbors(config.neighborhood,j,X)
      for i in neigh
          w = weight(config.oracle,i,j,X)
          connect!(g,i,j,w)
      end
  end
end
=#
