using SpectralClustering
using LightGraphs
using Base.Test
function weight_matrix(n_type::Type{T},g::SpectralClustering.Graph) where T<:AbstractFloat
    d=zeros(n_type,nv(g))
    if (g.is_dirty)
      reindex!(g);
    end
    number_of_edges = ne(g)
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
   W = sparse(I,J,V,nv(g),nv(g))
   return W
end
number_of_vertices = 5
g = SpectralClustering.Graph(number_of_vertices)
w = zeros(number_of_vertices, number_of_vertices)
edges = [(2,1, 1.0), (3,1,2.0), (4,1,3.0), (5,1,1.0), (2,3,2.0),(3,5,3.0),(4,2,4.0)]
for e in edges
    connect!(g, e[1], e[2], e[3])
    w[e[1],e[2]]=e[3]
    w[e[2],e[1]]=e[3]
end
@test isequal(adjacency_matrix(g),sparse(w))
@test isequal(adjacency_matrix(g),weight_matrix(Float64,g))
# write your own tests here
#TODO TESTS!
