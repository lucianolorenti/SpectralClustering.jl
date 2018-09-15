export
       Creation,
       Matrices,
       Graph,
       connect!,
       disconnect,
       remove_vertex!,
       reindex!,
       random_graph,
       target_vertex,
       cycles,
    number_of_neighbors

using LightGraphs

import LightGraphs: nv, ne, has_edge, is_directed

import Base.start
import Base.next
import Base.done
import Base.length
import Base.empty!
"""
```julia
type Edge
```

"""
struct Edge{T}
  next_v1::Union{Nothing,Edge{T}}
  prev_v1::Union{Nothing,Edge{T}}
  next_v2::Union{Nothing,Edge{T}}
  prev_v2::Union{Nothing,Edge{T}}
  v1
  v2
  weight::T
end
function weight_type(edge::Edge{T}) where T
    return T
end
struct Vertex{T, EdgeType}
   id::Integer
   data::T
   edges::Union{Nothing,Edge{EdgeType}}
   number_of_edges::Integer
   degree::Float64
   connections::Set{Integer}
   lock::Threads.TatasLock
end
function weight_type(v::Vertex{V,E}) where V where E
    return E
end
function Vertex(id::Integer,d::DataType = Any, val=nothing, weight_type::DataType=Float64)
    return Vertex{d, weight_type}(id,val,nothing,0,0, Set{Integer}(),Threads.TatasLock())
end
function Edge(v1::Vertex,v2::Vertex,w::Number)
    return Edge{weight_type(v1)}(nothing,nothing,nothing,nothing,v1,v2,convert(weight_type(v1),w))
end
struct Graph
  vertices::Vector{Vertex}
  is_dirty::Bool
end
struct EdgeIterator
   e::Union{Nothing,Edge}
   i::Integer
end
function empty!(g::Graph)
  for i=1:length(g.vertices)
    g.vertices[i].edges = nothing
    g.vertices[i].number_of_edges = 0
    g.vertices[i].degree = 0
  end
end
"""
```julia
Graph(n_vertices::Integer=0; vertex_type::DataType  = Any ,initial_value=nothing, weight_type::DataType = Float64)
```
Construct an undirected weighted grpah of `n_vertices` vertices.
"""
function Graph(n_vertices::Integer=0; vertex_type::DataType  = Any ,initial_value=nothing, weight_type::DataType = Float64)
   vertices = Vector{Vertex{vertex_type, weight_type}}(n_vertices)
   for i=1:n_vertices
       vertices[i] = Vertex(i,vertex_type,initial_value, weight_type)
   end
   return Graph(vertices,false)
end


function is_directed(g::Graph)
    return false
end
"""
```julia
nv(g::Graph)
```
Return the number of vertices of `g`.
"""
function nv(g::Graph)
    return length(g.vertices)
end
function ne(g::Graph)
    return sum([v.number_of_edges for v in g.vertices])
end
function has_edge(gr::Graph, i::Integer, k::Integer)
    return (k in gr.vertices[i].connections)
end
function insert!(v::Vertex,e::Edge)
    if (e.v1 == v)
        #El siguiente del nuevo es el primero de la lista original
        e.next_v1 = v.edges
        #El anterior del primero de la lista original es ahora el nuevo
        if (v.edges != nothing)
          if (v.edges.v1 == v)
              v.edges.prev_v1 = e
         else
              v.edges.prev_v2 = e
          end
        end
        v.edges = e
    else
        e.next_v2 = v.edges
        if (v.edges != nothing)
          if (v.edges.v1 == v)
              v.edges.prev_v1 = e
          else
            v.edges.prev_v2 = e
          end
        end
        v.edges = e
    end
end

function set_previous(v::Vertex,e::Edge,prev::Union{Void,Edge})
    if (e.v1 == v)
        e.prev_v1=prev
    else
        e.prev_v2=prev
    end
end

function set_next(v::Vertex,e::Edge,next::Union{Void,Edge})
    if (e.v1 == v)
        e.next_v1=next
    else
        e.next_v2=next
    end
end
function linked_list_connect(v::Vertex,e::Edge,next::Union{Void,Edge})
    set_next(v,e,next)
    if (next != nothing)
        set_previous(v,next,e)
    end
end
function remove!(v::Vertex,e::Edge)
    if (e.v1 == v)
        prev = e.prev_v1
        if (prev == nothing)
           v.edges = e.next_v1
           if e.next_v1 != nothing
               set_previous(v,e.next_v1,nothing)
            end
           return
        else
           linked_list_connect(v,prev,e.next_v1)
        end
    else
        prev = e.prev_v2
        if (prev == nothing)
           v.edges = e.next_v2
           if e.next_v2 != nothing
              set_previous(v,e.next_v2,nothing)
           end
           return
        else
           linked_list_connect(v,prev,e.next_v2)
        end
    end
end
"""
```julia
function connect!(g::Graph, i::Integer, neighbors::Vector, weigths::Vector)
```
"""
function connect!(g::Graph, i::Integer, neighbors::Vector, w::Vector)
    for j=1:length(neighbors)
        connect!(g,i,neighbors[j],w[j])
    end
end

"""
```julia
connect!(g::Graph,i::Integer,j::Integer,w::Number)
```
Connect the vertex `i` with the vertex `j` with weight `w`.
"""
function connect!(g::Graph,i::Integer,j::Integer,w::Number)
    if (i==j)
        return
    end
    if i>nv(g) || j>nv(g)
      throw("Invalid vertex")
    end
    if (w<=0)
        return
    end
    vertex_j = g.vertices[j]
    vertex_i = g.vertices[i]

     if !(vertex_j.id in vertex_i.connections)
      edge = Edge(g.vertices[i],g.vertices[j],w)
      lock(vertex_i.lock)
      lock(vertex_j.lock)
      insert!(g.vertices[i],edge)
      insert!(g.vertices[j],edge)
      push!(vertex_i.connections, vertex_j.id)
      push!(vertex_j.connections, vertex_i.id)
      unlock(vertex_j.lock)
      unlock(vertex_i.lock)
      g.vertices[i].number_of_edges=g.vertices[i].number_of_edges+1
      g.vertices[j].number_of_edges=g.vertices[j].number_of_edges+1
      g.vertices[i].degree = g.vertices[i].degree + w
      g.vertices[j].degree = g.vertices[j].degree + w
    end

end

function _advance(e::Edge,v::Vertex,ei::EdgeIterator)
   ei.e = e
   if (e.v1 == v)
        ei.i=1
   else
        ei.i=2
   end
end
function _start(e::Edge,v::Vertex)
   ei = EdgeIterator(nothing,-1)
   if (ei!=nothing)
       _advance(e,v,ei)
   end
   return ei
end
"""
```julia
target_vertex(e::Edge,v::Vertex)
```
Given an edge `e` and a vertex `v` returns the other vertex different from `v`
"""
function target_vertex(e::Edge,v::Vertex)
    if (e.v1 == v)
        return e.v2
    elseif (e.v2 == v)
        return e.v1
    else
        return Nothing
    end
end
"""
```julia
length(v::Vertex)
```
Return the number of edges connected to a given vertex.
"""
function length(v::Vertex)
    return v.number_of_edges
end
function start(v::Vertex)
    if (v.edges!=nothing)
       return _start(v.edges,v)
    else
        return EdgeIterator(nothing,0)
    end
end
function done(v::Vertex,ei::EdgeIterator)
    return ei.e == nothing
end
function next(v::Vertex,ei::EdgeIterator)
    edge = ei.e
    if (ei.i == 1)
        edge_next = ei.e.next_v1
    else
        edge_next = ei.e.next_v2
    end
    if (edge_next != nothing)
        _advance(edge_next,v,ei)
    else
       ei.e=nothing
    end
    return (edge, ei)
end

import Base.show
function show(io::IO, e::Edge)
  println(string(e.v1.id," -(",e.weight, ")> ",e.v2.id))
end
function show(io::IO, g::Graph)
  for vertex in g.vertices
         println(vertex)
         for e in vertex
           println(e)
       end
   end
end

function show(io::IO, v::Vertex)
  println("Vertice" )
  println("id: $(v.id)" )
  println("Datos: $(v.data)" )
  println("Aristas: $(v.number_of_edges)" )
  println("Grado: $(v.degree)" )

end
function find_edge(p, v::Vertex)
     for e in v
         if (p(e))
           return e
         end
      end
      return nothing
end
"""
```julia
disconnect(g::Graph,i::Integer,j::Integer)
```
Removes the edge that connects the `i`-th vertex to the `j`-th vertex.
"""
function disconnect(g::Graph,i::Integer,j::Integer)
   vertex_i = g.vertices[i]
   vertex_j = g.vertices[j]
   edge = find_edge(e->(e.v1 == vertex_j || e.v2==vertex_j),g.vertices[i])
   remove!(vertex_i,edge)
   remove!(vertex_j,edge)
   delete!(vertex_i.connections, vertex_j.id)
   delete!(vertex_j.connections, vertex_i.id)
   vertex_i.degree = vertex_i.degree - edge.weight
   vertex_j.degree = vertex_j.degree - edge.weight
   g.is_dirty = true
end
function update_connections!(g::Graph)

   for i=1:nv(g)
       empty!(g.vertices[i].connections)
       for e in g.vertices[i]
            v_j = target_vertex(e, g.vertices[i])
            push!(g.vertices[i].connections, v_j.id)
       end
   end
end

function reindex!(g::Graph)
   for i=1:nv(g)
       g.vertices[i].id = i
   end
   update_connections!(g)
   g.is_dirty = false
end
"""
```julia
remove_vertex!(g::Graph,i::Integer)
```
Remove the `i`-th vertex.
"""
function remove_vertex!(g::Graph,i::Integer)
    if i>nv(g)
      throw("No se puede eliminar")
    end
    if (g.is_dirty)
      reindex!(g)
    end
    vertex_i = g.vertices[i]
    for e in vertex_i
      if e.v1 == vertex_i
            remove!(e.v2,e)
            e.v2.degree = e.v2.degree - e.weight
            e.v2.number_of_edges = e.v2.number_of_edges-1
        else
            remove!(e.v1,e)
            e.v1.number_of_edges = e.v1.number_of_edges-1
            e.v1.degree = e.v1.degree - e.weight
        end
    end
    deleteat!(g.vertices,i)
    g.is_dirty=true
end
function add_vertex!(g::Graph, datatype = Any, data=nothing)
   if g.is_dirty
      reindex!(g)
   end
   new_id = nv(g) +1
   vertex = Vertex(new_id,datatype, data)
   push!(g.vertices,vertex)
   return vertex
end
function connect(e::Edge, v::Vertex)
   return e.v1 == v || e.v2 == v
end
function number_of_neighbors(g::Graph)
    number = zeros(Int,nv(g))
    for i=1:length(g.vertices)
        n=0
        for e in g.vertices[i]
            n= n +1
        end
        number[i] = n
    end
    return number

end
"""
```
function random_graph(iterations::Integer; probs=[0.4,0.4,0.2], weight=()->5, debug=false)
```
Create a random graphs. `probs` is an array of probabilities. The function create a vertex with probability `probs[1]`, connect two vertices with probability `probs[2]` and delete a vertex with probability `probs[2]`. The weight of the edges is given by `weight`
"""
function random_graph(iterations::Integer; probs=[0.4,0.4,0.2], weight=()->5, debug=false)
  g= Graph()
  for i=1:iterations
    action = sum(rand() .>= cumsum(probs)) +1
    nog = nv(g)
    if action==1
        add_vertex!(g)
        if debug
           println("add_vertex!(g)")
        end
    elseif action == 2
        if (nog >= 2)
           v1 = rand(1:nog)
           v2 = rand(1:nog)
           if (v1!=v2)
             if (debug)
               println("connect!(g,$(v1),$(v2),5)")
             end
             connect!(g,v1,v2,weight())
           end
        end
     elseif action == 3
       if (nog > 0)
          v1 = rand(1:nog)
          if (debug)
            println("remove_vertex!(g,$(v1))")
          end
          remove_vertex!(g,v1)
        end
     end
  end
  return g
end
struct TargetVertexAndWeight
    vertex_id::Integer
    edge_weight::Float64
end
struct Triangle
    edge_1::TargetVertexAndWeight
    edge_2::TargetVertexAndWeight
end
function Triangle()
    return Triangle((-1,-1.0),(-1,-1.0))
end
import Base.hash
import Base.isequal

function hash(a::Triangle, h::UInt)
    if a.edge_1.vertex_id < a.edge_2.vertex_id
        hash(a.edge_1.vertex_id, hash(a.edge_2.vertex_id, hash(:Triangle, h)))
    else
        hash(a.edge_2.vertex_id, hash(a.edge_1.vertex_id, hash(:Triangle, h)))
    end

end
isequal(a::Triangle, b::Triangle) = Base.isequal(hash(a), hash(b))
function compute_cycles(vertice_inicio::Vertex, vertice::Vertex,
                                        visitados::Vector{Bool}, ciclos::Set{Triangle},
                                        cant_aristas::Integer, peso_aristas::TargetVertexAndWeight)
   if  !visitados[vertice.id]
     visitados[vertice.id] = true
     cant_aristas = cant_aristas+1
     for edge in vertice
          vertice_destino = target_vertex(edge, vertice)
          if cant_aristas == 1
               peso_aristas.vertex_id  = vertice_destino.id
               peso_aristas.edge_weight  = edge.weight
          end
          if vertice_destino == vertice_inicio && cant_aristas == 3

               push!(ciclos, Triangle( deepcopy(peso_aristas), TargetVertexAndWeight(vertice.id, edge.weight)))
               visitados[vertice.id]=false
               return  #Si encontre un ciclo, no voy a encontrar otro
          else
               if !visitados[vertice_destino.id]
                    if cant_aristas < 3
                         compute_cycles(vertice_inicio, vertice_destino, visitados, ciclos,
                                        cant_aristas,peso_aristas)

                    end
               end
          end
     end
     visitados[vertice.id]=false
   end

end

function cycles(g::Graph)
    visitados = fill(false,length(g.vertices))
    ciclos_vertices = []
    for i=1:length(g.vertices)
        v = g.vertices[i]
        peso_aristas = TargetVertexAndWeight(-1,-1.0)
        ciclos = Set{Triangle}()
        compute_cycles(v,v, visitados,ciclos,0,peso_aristas)
        push!(ciclos_vertices,ciclos)
    end
    return ciclos_vertices
end

include("Creation.jl")
include("Matrices.jl")
include("Plot.jl")

#=Example
g = Graph(11)
connect!(g,3,2,5)
connect!(g,2,1,5)
remove_vertex!(g,1)
add_vertex!(g)
add_vertex!(g)
remove_vertex!(g,3)
remove_vertex!(g,2)
=#
