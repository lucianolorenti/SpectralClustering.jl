import LightGraphs.LinAlg: adjacency_matrix
"""
    adjacency_matrix(g[, T=Int; dir=:out])
Return a sparse adjacency matrix for a graph, indexed by `[u, v]`
vertices. Non-zero values indicate an edge between `u` and `v`. Users may
override the default data type (`Int`) and specify an optional direction.
### Optional Arguments
`dir=:out`: `:in`, `:out`, or `:both` are currently supported.
### Implementation Notes
This function is optimized for speed and directly manipulates CSC sparse matrix fields.
"""
function adjacency_matrix(g::Graph, T::DataType=Float64; dir::Symbol=:out)
   n_v = nv(g)
   nz = ne(g)
   colpt = ones(Int64, n_v + 1)

   rowval = sizehint!(Vector{Int64}(), nz)
   weights   = sizehint!(Vector{T}(), nz)
    for j in 1:n_v  # this is by column, not by row.
       wgts = sizehint!(Vector{T}(), g.vertices[j].number_of_edges)
       dsts = sizehint!(Vector{Int64}(), g.vertices[j].number_of_edges)
        for e in g.vertices[j]
            push!(wgts,e.weight)
            push!(dsts,target_vertex(e,g.vertices[j]).id)
        end
        colpt[j + 1] = colpt[j] + length(dsts)

       dsts_indices = sortperm(dsts)
        append!(rowval, dsts[dsts_indices])
        append!(weights, wgts[dsts_indices])
    end
    return SparseMatrixCSC(n_v, n_v, colpt, rowval, weights)

end
