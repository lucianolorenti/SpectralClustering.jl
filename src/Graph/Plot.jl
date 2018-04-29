export plot
using Plots
"""
```julia
    plot(g::Graph, X::Matrix)
```
Plots de graph ```g``` in a 2D space using X as coordinates for the vertex
"""
function plot(g::Graph, X::Matrix)
  plt = scatter(X[1,:],X[2,:],legend=nothing)
  for v in g.vertices
    for e in v
      i = e.v1.id
      j = e.v2.id
      Plots.plot!(plt,X[1,[i,j]],X[2,[i,j]],linealpha=0.50,legend=nothing, linewidth = e.weight*4 ,linecolor=RGB(1.0,0.2,0))
    end
  end
  return plt
end
