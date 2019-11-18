module SpectralClustering
export EigenvectorEmbedder,
       embedding

abstract type EigenvectorEmbedder end

include("Utils/DataAccess.jl")
include("Utils/DataProcessing.jl")
include("Graph/Graphs.jl")
include("LandmarkSelection.jl")
include("Embedding.jl")
include("ApproximateEmbedding.jl")
include("MultiView.jl")
include("EigenvectorClustering.jl")
end
