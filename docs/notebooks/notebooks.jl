include(joinpath(dirname(@__FILE__),"..","deps.jl"))
push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../"))
using DocUtils
using IJulia
notebook(dir=dirname(@__FILE__))
