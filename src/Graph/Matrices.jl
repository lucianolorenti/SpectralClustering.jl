export degrees,
       weight_matrix,
       ng_laplacian,
       laplacian,
       incidence,
       normalized_laplacian,
       shifted_laplacian
"""
    degrees(g::Graph)

Returns a vector with the degrees of every vertex
"""
function degrees(g::Graph)
  return [v.degree for v in g.vertices]
end

"""
```julia
weight_matrix(g::Graph)
```

Given a graph ´g´ returns the weight matrix and a vector with the
vertices degrees.
"""
function weight_matrix(g::Graph)
    return weight_matrix(Float64, g);
end
"""
```julia
weight_matrix(n_type::T,g::Graph) where T<:AbstractFloat
````

Given a graph ´g´ returns the weight matrix of ``n_type` and a vector
with the vertices degrees.
"""
function weight_matrix(n_type::Type{T},g::Graph) where T<AbstractFloat
    d=zeros(n_type,number_of_vertices(g))
    if (g.is_dirty)
      reindex!(g);
    end
    number_of_edges = 0
    for v in g.vertices
        number_of_edges = number_of_edges + v.number_of_edges
    end

    I = zeros(Int,number_of_edges)
    J = zeros(Int,number_of_edges)
    V = zeros(n_type,number_of_edges)
    j= 1
    for v in g.vertices
      d[v.id] = v.degree
      for e in v
        I[j] = v.id
        J[j] = target_vertex(e,v).id
        V[j] = e.weight
        j=j+1

      end
   end
   W = sparse(I,J,V,number_of_vertices(g),number_of_vertices(g))
   return (W,d)
end

"""
    weight_matrix(g::Graph)

Given a graph ´´´g´´´ and two array of vertices (set1, set2) indexes returns the weight matrix form the vertices
in set1 to the vertices in set2.
"""
function weight_matrix(g::Graph, set1::Vector{Int},set2::Vector{Int})
  if (g.is_dirty)
      reindex!(g)
  end
  m = zeros(length(set1),length(set2))
  for j=1:length(set1)
    vertex_j = g.vertices[set1[j]]
    for edge in vertex_j
      for k=1:length(set2)
         vertex_k = g.vertices[set2[k]]
         if connect(edge,vertex_k)
             m[j,k]=edge.weight
          end
       end
    end
  end
  return sparse(m)
end
"""
    incidence(g::Graph)

Returns the incidence matrix of the graph ´´´G´´´
"""
function incidence(g::Graph)
  check_and_correct(g)
  m=zeros(number_of_vertices(g),number_of_vertices(g))
  for vertex in g.vertices
    for edge in vertex
         v1_row = edge.v1.id
         v2_row = edge.v2.id
         m[v1_row,v2_row] = 1
          m[v2_row,v1_row] = 1
    end
  end
  return m
end
"""
    ng_laplacian(g::Graph)

Returns the laplacian matrix of the graph ´´´G´´´ according to ""On Spectral Clustering: Analysis and
an algorithm"" of Andrew Y. Ng, Michael I. Jordan and Yair Weiss.

\$L = D^{-\\frac{1}{2}} W D^{-\\frac{1}{2}}\$

"""
function ng_laplacian(g::Graph)
    (W,d) = weight_matrix(g)
    d= vec(d)
    d[d.<0]= 0
    d[d.!=0] =d[d.!=0].^-0.5
    Dinv = spdiagm(d)
    return (Dinv*W*Dinv, Dinv)
end
function laplacian(W::SparseMatrixCSC)
  d = vec(sum(W,1))
  return laplacian(W,d)
end

"""
```
laplacian(W::SparseMatrixCSC, d::Vector)
```
Returns the laplacian matrix ´´´L´´´ of the graph ´´´g´´´
"""
function laplacian(W::SparseMatrixCSC, d::Vector)
    d[d.<0]= 0
    D = spdiagm(d)
    return D-W
end
"""
```
laplacian(g::Graph)
```

Returns the laplacian matrix ´´´L´´´ of the graph ´´´g´´´

\$L=D-W\$
"""
function laplacian(g::Graph)
    (W,d) = weight_matrix(g)
    return laplacian(W,d)
end
"""
```julia
 normalized_laplacian(W::SparseMatrixCSC)
```
"""
function normalized_laplacian(W::SparseMatrixCSC)
      d=vec(sum(W,1))
      return normalized_laplacian(W,d)
end
"""
```julia
normalized_laplacian(W::SparseMatrixCSC,d)
```
"""
function normalized_laplacian(W::SparseMatrixCSC,d)
      d[d.<0]= 0
      D = spdiagm(d)

      d1 = copy(d)
      d1[d1.!=0] =d1[d1.!=0].^0.5
      D1 = spdiagm(d1)
      d[d.!=0] =d[d.!=0].^-0.5

      Dinv = spdiagm(d)
      Lhat = Dinv*(D-W)*Dinv
      return (Lhat,Dinv)
end
"""
```julia
normalized_laplacian(g::Graph)
```

If W it is the weight matrix \$W\$ of ´´´g´´´, \$D=W \mathbf{1}\$. The normalized laplacian \$L = D^{-\\frac{1}{2}} (D-W) D^{-\\frac{1}{2}}\$. The function returns the normalized laplacian of the graph ´´´g´´´ and \$D^{\\frac{1}{2}}\$.
"""
function normalized_laplacian(g::Graph)
       (W,d) = weight_matrix(g)
       return normalized_laplacian(W,d)
end

function shifted_laplacian(g::Graph)
  (W,d) = weight_matrix(g)
  d= vec(d)
  d[d.<0]= 0
  d[d.!=0] =1./sqrt(d[d.!=0])
  D = spdiagm(d)
  #i = zeros(size(W,1))
  #i[d.!=0] = 1
  #I = spdiagm(i)
  Lhat = eye(size(W,1)) + (D*W*D)
  return Lhat
end


