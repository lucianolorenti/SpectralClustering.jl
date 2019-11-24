push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../"))
using IJulia
notebook(dir=dirname(@__FILE__))
