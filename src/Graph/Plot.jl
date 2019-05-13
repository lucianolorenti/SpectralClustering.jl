using Plots
using GraphRecipes
import GraphRecipes: get_source_destiny_weight
function get_source_destiny_weight(g::Graph)
    L = ne(g)
    sources = Array{Int}(undef, L)
    destiny = Array{Int}(undef, L)
    weights = Array{Float64}(undef, L)
    i = 0
    for v in g.vertices
        for e in v
            i += 1
            sources[i] = e.v1.id
            destiny[i] = e.v2.id
            weights[i] = e.weight
        end
    end
    return sources, destiny, weights
end
