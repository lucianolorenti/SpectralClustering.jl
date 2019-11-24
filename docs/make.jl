using Documenter
using SpectralClustering

deploydocs(
    repo = "github.com/lucianolorenti/SpectralClustering.jl.git",
    julia  = "1.2.0",
    deps = nothing,
    make = nothing,
    target = "build"
)



makedocs(
    modules = [SpectralClustering],
    format = Documenter.HTML(prettyurls = true),
    source = "src",
    clean = false,
    
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
mkpath(notebook_output_dir)
using IJulia
jupyter_path = first(IJulia.find_jupyter_subcommand("nbconvert"))
for file in readdir(joinpath(dirname(@__FILE__), "notebooks"))
   full_path = joinpath(dirname(@__FILE__), "notebooks", file)
    if (endswith(file,".ipynb"))
	    run(`$(jupyter_path) nbconvert --to html $full_path --output-dir=$notebook_output_dir`)
    elseif (file != ".ipynb_checkpoints")
        cp(full_path, joinpath(notebook_output_dir,file),  force=true)
    end
end