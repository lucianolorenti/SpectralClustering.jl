push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../"))
using DocUtils
using IJulia
notebook(dir=dirname(@__FILE__))
