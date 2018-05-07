using SpectralClustering
using LightGraphs
using Base.Test

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
# write your own tests here
#TODO TESTS!
