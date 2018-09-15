include(joinpath(dirname(@__FILE__),"deps.jl"))
push!(LOAD_PATH, dirname(@__FILE__))
using DocUtils
using Extras
makedocs(
     modules = [SpectralClustering],
    format = :html,
    source = "src",
    assets = [ "assets/css/custom.css", "assets/js/mathjaxhelper.js"],
    sitename = "SpectralClustering.jl",
    pages = Any[
        "Home" => "index.md",
        "Getting Started"=>"start.md",
         "Main Modules" => Any[
                       "Graph Creation" => "man/graphcreation.md",
                       "Embedding"      => "man/embedding.md",
                       "Approximate Embedding" => "man/approximate.md",
                       "Eigenvector Clustering" => "man/clusterize.md",
                       "Co-Regularized" => "man/multiview.md",
                       "Incremental"    => "man/incremental.md"
         ],
        "Utility Modules" => Any[
                       "Data Access" => "man/data_access.md",
                       "Graph" => "man/graph.md",
                       "Landmarks Selection" => "man/landmark_selection.md"
        ],
   ],
  doctest = false

)
notebook_output_dir =   joinpath(dirname(@__FILE__), "build","notebooks")
using IJulia
for file in readdir(joinpath(dirname(@__FILE__), "notebooks"))
   full_path = joinpath(dirname(@__FILE__), "notebooks", file)
    if (endswith(file,".ipynb"))
	run(`$(IJulia.jupyter) nbconvert --template=nbextensions --to html $full_path --output-dir=$notebook_output_dir`)
    else
        cp(full_path, joinpath(notebook_output_dir,file))
    end
end
deploydocs(
    repo = "github.com/lucianolorenti/SpectralClustering.jl.git",
    julia  = "0.6",
    deps = nothing,
    make = nothing,
    target = "build"
)
