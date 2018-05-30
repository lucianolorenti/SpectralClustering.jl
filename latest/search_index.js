var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#SpectralClustering.jl-1",
    "page": "Home",
    "title": "SpectralClustering.jl",
    "category": "section",
    "text": "Given a set of patterns X=x_1x_2x_n in mathbb R^m, and a simmilarity function  dmathbb R^m times mathbb R^m  rightarrow mathbb R, is possible to build an affinity matrix W such that  W(ij) = d(x_i x_j). Spectral clustering algorithms obtains a low rank representation of the patterns solving the following optimization problembeginarrayccc\nmax  mboxTr(U^T L  U) \nU in mathbb R^ntimes k  \ntextrmsa  U^T U  =   I\nendarraywhere L = D^-frac12WD^-frac12 is the Laplacian matrix derived from W according ng2002spectral and D is a diagonal matrix with the sum of the rows of W located in its main diagonal. Once obtained U, their rows are considered as the new coordinates of the patterns. In this new representation is simpler to apply a traditional clustering algorithm  shi2000normalized.Spectral graph partitioning methods have been successfully applied to circuit layout [3, 1], load balancing [4] and image segmentation [10, 6]. As a discriminative approach, they do not make assumptions about the global structure of data. Instead, local evidence on how likely two data points belong to the same class is first collected and a global decision is then made to divide all data points into disjunct sets according to some criterion. Often, such a criterion can be interpreted in an embedding framework, where the grouping relationships among data points are preserved as much as possible in a lower-dimensional representation."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "At the Julia REPL:Pkg.clone(\"https://github.com/lucianolorenti/SpectralClustering.jl.git\")"
},

{
    "location": "index.html#Description-1",
    "page": "Home",
    "title": "Description",
    "category": "section",
    "text": "The library provides functions that allow:Build the affinity matrix. Simmilarity graph creation, Graph matrices\nPerform the embedding of the patterns in the space spanned by the eigenvectors of the matrices derived from the affinity matrix. Eigenvector Embedding\nObtain an approximation of the eigenvector in order to reduce the computational complexity. Approximate embedding\nExploiting information from multiple views. Corresponding nodes in each graph should have the same cluster membership. MultiView embedding\nClusterize the eigenvector space. Eigenvector Clustering"
},

{
    "location": "index.html#Bibliography-1",
    "page": "Home",
    "title": "Bibliography",
    "category": "section",
    "text": "import Documenter.Documents.RawHTML\nusing DocUtils\nRawHTML(bibliography([\"ng2002spectral\",\"shi2000normalized\",\"yu2001understanding\"]))"
},

{
    "location": "start.html#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "start.html#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphcreation.html#",
    "page": "Graph Creation",
    "title": "Graph Creation",
    "category": "page",
    "text": ""
},

{
    "location": "man/graphcreation.html#Simmilarity-graph-creation-1",
    "page": "Graph Creation",
    "title": "Simmilarity graph creation",
    "category": "section",
    "text": "A weighted graph is an ordered pair G=(VE) that is composed of a set V of vertices together with a set E of edges (ijw) ij in Vw in R. The number w, the weight, represent the simmilarity between i and j.In order to build a simmilarity graph two elements have to be defined:Which are the neighbors for a given vertex. For this, a concrete type that inherit from NeighborhoodConfig  has to be instantiated. \nThe simmilarity function between patterns.  The function receives the element being evaluated and its neighbors and returns a vector with the simmilarities between them.  The signature of the function has to be the following function weight(i::Integer, j::Vector{Integer}, e1, e2) where i::Int is the index of the pattern being evaluated, j::Vector{Integer}  are the indices of the neighbors of i;  e1 are the i-th pattern and  e2 are the  neighbors patterns."
},

{
    "location": "man/graphcreation.html#Examples-1",
    "page": "Graph Creation",
    "title": "Examples",
    "category": "section",
    "text": "Graph creation examples"
},

{
    "location": "man/graphcreation.html#Bibliography-1",
    "page": "Graph Creation",
    "title": "Bibliography",
    "category": "section",
    "text": "import Documenter.Documents.RawHTML\nusing DocUtils\nRawHTML(bibliography([\"Zelnik-manor04self-tuningspectral\"]))"
},

{
    "location": "man/graphcreation.html#Reference-1",
    "page": "Graph Creation",
    "title": "Reference",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphcreation.html#Index-1",
    "page": "Graph Creation",
    "title": "Index",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages   = [\"man/graphcreation.md\"]"
},

{
    "location": "man/graphcreation.html#SpectralClustering.KNNNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.KNNNeighborhood",
    "category": "type",
    "text": "type KNNNeighborhood <: VertexNeighborhood\n    k::Integer\n    tree::KDTree\nend\n\nKNNNeighborhood specifies that the neighborhood for a given vertex j are the k nearest neighborgs. It uses a tree to search the nearest patterns.\n\nMembers\n\nk::Integer. The number of k nearest neighborgs to connect.\ntree::KDTree. Internal data structure.\nf::Function. Transformation function\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.KNNNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.KNNNeighborhood",
    "category": "type",
    "text": "KNNNeighborhood(X::Matrix, k::Integer)\n\nCreate the KNNNeighborhood type by building a k-nn tre from de data X\n\nReturn the indexes of the config.k nearest neigbors of the data point j of the data X.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.PixelNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.PixelNeighborhood",
    "category": "type",
    "text": "type PixelNeighborhood  <: VertexNeighborhood\n\nPixelNeighborhood defines neighborhood for a given pixel based in its spatial location. Given a pixel located at (x,y), returns every pixel inside (x+ey) (x-ey) and (xy+e)(xy-e).\n\nMembers\n\ne:: Integer. Defines the radius of the neighborhood.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.RandomNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.RandomNeighborhood",
    "category": "type",
    "text": "type RandomNeighborhood <: VertexNeighborhood\n    k::Integer\nend\n\nFor a given index jreturn k random vertices different from j\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.VertexNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.VertexNeighborhood",
    "category": "type",
    "text": "abstract type VertexNeighborhood end\n\nThe abstract type VertexNeighborhood provides an interface to query for the neighborhood of a given vertex. Every concrete type that inherit from VertexNeighborhood must define the function\n\nneighbors{T<:VertexNeighborhood}(cfg::T, j::Integer, data)\n\nwhich returns the neighbors list of the vertex j for the given data.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.create-Tuple{DataType,SpectralClustering.VertexNeighborhood,Function,Any}",
    "page": "Graph Creation",
    "title": "SpectralClustering.create",
    "category": "method",
    "text": "create(w_type::DataType, neighborhood::VertexNeighborhood, oracle::Function,X)\n\nGiven a VertexNeighborhood, a simmilarity function oracle  construct a simmilarity graph of the patterns in X.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.create-Tuple{SpectralClustering.RandomKGraph}",
    "page": "Graph Creation",
    "title": "SpectralClustering.create",
    "category": "method",
    "text": "create(cfg::RandomKGraph)\n\nConstruct a RandomKGraph such that every vertex is connected with other k random vertices.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.create-Tuple{SpectralClustering.VertexNeighborhood,Function,Any}",
    "page": "Graph Creation",
    "title": "SpectralClustering.create",
    "category": "method",
    "text": "create(neighborhood::VertexNeighborhood, oracle::Function,X)\n\nGiven a VertexNeighborhood, a simmilarity function oracle construct a simmilarity graph of the patterns in X.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.local_scale-Tuple{SpectralClustering.KNNNeighborhood,Function,Any}",
    "page": "Graph Creation",
    "title": "SpectralClustering.local_scale",
    "category": "method",
    "text": "local_scale(neighborhood::KNNNeighborhood, oracle::Function, X; k = 7)\n\nComputes the local scale of each pattern according to Self-Tuning Spectral Clustering. Return a matrix containing for every pattern the local_scale.\n\n\"The selection of the local scale $ \\sigma $ can be done by studying the local statistics of the neighborhoods surrounding points $ i $ and $ j $ .i \" Zelnik-Manor and Perona use $ \\sigma_i = d(s_i, s_K) $ where s_K is the $ K $ neighbor of point $ s_i $ . They \"used a single value of K=7, which gave good results even for high-dimensional data \" .\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.neighbors-Tuple{SpectralClustering.CliqueNeighborhood,Integer,Any}",
    "page": "Graph Creation",
    "title": "SpectralClustering.neighbors",
    "category": "method",
    "text": "neighbors(config::CliqueNeighborhood, j::Integer, X)\n\nReturn every other vertex index different from j. See CliqueNeighborhood\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.neighbors-Union{Tuple{SpectralClustering.PixelNeighborhood,Integer,Array{T,2}}, Tuple{T}} where T<:ColorTypes.Colorant",
    "page": "Graph Creation",
    "title": "SpectralClustering.neighbors",
    "category": "method",
    "text": "neighbors(cfg::PixelNeighborhood, j::Integer, img)\n\nReturns the neighbors of the pixel j according to the specified in PixelNeighborhood\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.CliqueNeighborhood",
    "page": "Graph Creation",
    "title": "SpectralClustering.CliqueNeighborhood",
    "category": "type",
    "text": "type CliqueNeighborhood <: VertexNeighborhood\n\nCliqueNeighborhood specifies that the neighborhood for a given vertex j in a graph of n vertices are the remaining n-1 vertices.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.RandomKGraph",
    "page": "Graph Creation",
    "title": "SpectralClustering.RandomKGraph",
    "category": "type",
    "text": "type RandomKGraph\n\nThe type RandomKGraph defines the parameters needed to create a random k-graph. Every vertex it is connected to k random neigbors.\n\nMembers\n\nnumber_of_vertices::Integer. Defines the number of vertices of the graph.\nk::Integer. Defines the minimum number of  neighborhood of every vertex.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#SpectralClustering.weight-Tuple{Function,Integer,Integer,Any}",
    "page": "Graph Creation",
    "title": "SpectralClustering.weight",
    "category": "method",
    "text": "weight{T<:DataAccessor}(w::Function,d::T, i::Int,j::Int,X)\n\nInvoke the weight function provided to compute the similarity between the pattern i and the pattern j.\n\n\n\n"
},

{
    "location": "man/graphcreation.html#Content-1",
    "page": "Graph Creation",
    "title": "Content",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"Graph/Creation.jl\"]"
},

{
    "location": "man/embedding.html#",
    "page": "Embedding",
    "title": "Embedding",
    "category": "page",
    "text": ""
},

{
    "location": "man/embedding.html#Eigenvector-Embedding-1",
    "page": "Embedding",
    "title": "Eigenvector Embedding",
    "category": "section",
    "text": "Spectral clustering techniques require the computation of the extreme eigenvectors of matrices derived from patterns similarity . The Laplacian matrix obtained from the data is generally used as the starting point for decomposition into autovectors. Given the symmetric matrix $ W (i, j) = w_{ij}, W \\in R^{n \\times n} $ that contains information about  similarity between the patterns, if $ D = W \\mathbf {1} $, the unnormalized Laplacian matrix is defined as $ L = D-W $.The matrix $ W $ can be seen as the incidence matrix of a weighted graph. The Graph Construction utilities implement functions that allow the construction of simmilarty graphs.The Embedding utilities contain the functions for performing the embedding of the patterns in the space spanned by the $ k $ eigenvectors of a matrix derived from W.Currently the module implements the techniques described in:On spectral clustering: Analysis and an algorithm.\nNormalized cuts and image segmentation.\nUnderstanding Popout through Repulsion.\nSegmentation Given Partial Grouping Constraints"
},

{
    "location": "man/embedding.html#Examples-1",
    "page": "Embedding",
    "title": "Examples",
    "category": "section",
    "text": "Embedding examples"
},

{
    "location": "man/embedding.html#Bibliography-1",
    "page": "Embedding",
    "title": "Bibliography",
    "category": "section",
    "text": "import Documenter.Documents.RawHTML\nusing DocUtils\nRawHTML(bibliography([\"ng2002spectral\",\"shi2000normalized\",\"yu2001understanding\", \"yu2004segmentation\",\"lee2007trajectory\"]))"
},

{
    "location": "man/embedding.html#Index-1",
    "page": "Embedding",
    "title": "Index",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"man/embedding.md\"]"
},

{
    "location": "man/embedding.html#SpectralClustering.NgLaplacian",
    "page": "Embedding",
    "title": "SpectralClustering.NgLaplacian",
    "category": "type",
    "text": "type NgLaplacian <: AbstractEmbedding\n\nMembers\n\nnev::Integer. The number of eigenvectors to obtain\n\nGiven a affinity matrix $ W \\in \\mathbb{R}^{n \\times n} $.  Ng et al defines the laplacian as $ L =  D^{-\\frac{1}{2}} W D^{-\\frac{1}{2}} $ where $ D $ is a diagonal matrix whose (i,i)-element is the sum of W\'s i-th row.\n\nThe embedding function solves a relaxed version of the following optimization problem: beginarraycrclcl     displaystyle max_ U in mathbbR^ntimes k hspace10pt   mathrmTr(U^T L  U)  \n   textrmsa  U^T U  =   I  endarray\n\nU is a matrix that contains the nev  largest eigevectors of $ L $.\n\nReferences\n\nOn Spectral Clustering: Analysis and an algorithm. Andrew Y. Ng, Michael I. Jordan, Yair Weiss\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.PartialGroupingConstraints",
    "page": "Embedding",
    "title": "SpectralClustering.PartialGroupingConstraints",
    "category": "type",
    "text": "struct PartialGroupingConstraints <: AbstractEmbedding\n\nMembers\n\nnev::Integer. The number of eigenvector to obtain.\n\nSegmentation Given Partial Grouping Constraints Stella X. Yu and Jianbo Shi\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.ShiMalikLaplacian",
    "page": "Embedding",
    "title": "SpectralClustering.ShiMalikLaplacian",
    "category": "type",
    "text": "The normalized laplacian as defined in  $ D^{-\\frac{1}{2}} (D-W) D^{-\\frac{1}{2}} $.\n\nReferences:\n\nSpectral Graph Theory. Fan Chung\nNormalized Cuts and Image Segmentation. Jiambo Shi and Jitendra Malik\n\ntype ShiMalikLaplacian <: AbstractEmbedding\n\nMembers\n\nnev::Integer. The number of eigenvector to obtain.\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.YuShiPopout",
    "page": "Embedding",
    "title": "SpectralClustering.YuShiPopout",
    "category": "type",
    "text": "struct YuShiPopout <: AbstractEmbedding\n\nMembers\n\nnev::Integer. The number of eigenvector to obtain.\n\nUnderstanding Popout through Repulsion Stella X. Yu and Jianbo Shi\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Tuple{SpectralClustering.NgLaplacian,SpectralClustering.Graph}",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::NgLaplacian, gr::Graph)\n\nPerforms the eigendecomposition of the matrix $ L $ derived from the graph gr. The matrix $ L $ is defined according to NgLaplacian\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Tuple{SpectralClustering.NgLaplacian,Union{Array{T,2} where T, SparseMatrixCSC}}",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::NgLaplacian, L::Union{Matrix,SparseMatrixCSC})\n\nPerforms the eigendecomposition of the matrix $ L $ defined according to NgLaplacian\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Tuple{SpectralClustering.ShiMalikLaplacian,Union{SparseMatrixCSC, SpectralClustering.Graph}}",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::ShiMalikLaplacian, gr::Union{Graph,SparseMatrixCSC})\n\nParameters\n\ncfg::ShiMalikLaplacian. An instance of a ShiMalikLaplacian  that specify the number of eigenvectors to obtain\ngr::Union{Graph,SparseMatrixCSC}. The Graph(@ref Graph) or the weight matrix of wich is going to be computed the normalized laplacian matrix.\n\nPerforms the eigendecomposition of the normalized laplacian matrix of the graph gr defined acoording to ShiMalikLaplacian. Returns the cfg.nev eigenvectors associated with the non-zero smallest eigenvalues.\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Tuple{SpectralClustering.YuShiPopout,SpectralClustering.Graph,SpectralClustering.Graph,Array{Array{Integer,1},1}}",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "function embedding(cfg::YuShiPopout,  grA::Graph, grR::Graph)\n\nReferences\n\nGrouping with Directed Relationships. Stella X. Yu and Jianbo Shi\nUnderstanding Popout through Repulsion. Stella X. Yu and Jianbo Shi\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Tuple{SpectralClustering.YuShiPopout,SpectralClustering.Graph,SpectralClustering.Graph}",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "function embedding(cfg::YuShiPopout,  grA::Graph, grR::Graph)\n\nReferences\n\nGrouping with Directed Relationships. Stella X. Yu and Jianbo Shi\nUnderstanding Popout through Repulsion. Stella X. Yu and Jianbo Shi\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.embedding-Union{Tuple{T,SpectralClustering.VertexNeighborhood,Function,Any}, Tuple{T}} where T<:SpectralClustering.AbstractEmbedding",
    "page": "Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding{T<:AbstractEmbedding}(cfg::T, neighborhood::VertexNeighborhood, oracle::Function, data)\n\n\n\n"
},

{
    "location": "man/embedding.html#SpectralClustering.PGCMatrix",
    "page": "Embedding",
    "title": "SpectralClustering.PGCMatrix",
    "category": "type",
    "text": "struct PGCMatrix{T,I,F} <: AbstractMatrix{T}\n\nPartial grouping constraint structure. This sturct is passed to eigs to performe the L*x computation according to (41), (42) and (43) of \"\"Segmentation Given Partial Grouping Constraints\"\"\n\n\n\n"
},

{
    "location": "man/embedding.html#Content-1",
    "page": "Embedding",
    "title": "Content",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"src/Embedding.jl\"]"
},

{
    "location": "man/approximate.html#",
    "page": "Approximate Embedding",
    "title": "Approximate Embedding",
    "category": "page",
    "text": ""
},

{
    "location": "man/approximate.html#Approximate-embedding-1",
    "page": "Approximate Embedding",
    "title": "Approximate embedding",
    "category": "section",
    "text": "Given a symmetric affinity matrix $A$, we would like to compute the $k$ smallest eigenvectors of the Laplacian of A. Directly computing such eigenvectors can be very costly even with sophisticated solvers, due to the large size of $A$."
},

{
    "location": "man/approximate.html#Examples-1",
    "page": "Approximate Embedding",
    "title": "Examples",
    "category": "section",
    "text": "Approximate embedding examples"
},

{
    "location": "man/approximate.html#Bibliography-1",
    "page": "Approximate Embedding",
    "title": "Bibliography",
    "category": "section",
    "text": "import Documenter.Documents.RawHTML\nusing DocUtils\nRawHTML(bibliography([\"pont2017multiscale\"]))"
},

{
    "location": "man/approximate.html#Reference-1",
    "page": "Approximate Embedding",
    "title": "Reference",
    "category": "section",
    "text": ""
},

{
    "location": "man/approximate.html#Index-1",
    "page": "Approximate Embedding",
    "title": "Index",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"man/approximate.md\"]"
},

{
    "location": "man/approximate.html#SpectralClustering.DNCuts",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.DNCuts",
    "category": "type",
    "text": "type DNCuts\n\nMultiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation\n\nJordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Member, Ferran Marques, Jitendra Malik\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.LandmarkBasedRepresentation",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.LandmarkBasedRepresentation",
    "category": "type",
    "text": "Large Scale Spectral Clustering with Landmark-Based Representation Xinl ei Chen Deng Cai\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.NystromMethod",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.NystromMethod",
    "category": "type",
    "text": "\ntype NystromMethod{T<:AbstractLandmarkSelection}\nlandmarks_selector::T\nnumber_of_landmarks::Integer\nw::Function\nnvec::Integer\nend\n\nThe type NystromMethod proposed in  Spectral Grouping Using the Nystrom Method by Charless Fowlkes, Serge Belongie, Fan Chung, and Jitendra Malik. It has to be defined:\n\nlandmarks_selector::T<:AbstractLandmarkSelection. A mechanism to select the sampled\n\npoints.\n\nnumber_of_landmarks::Integer. The number of points to sample\nw::Function. The weight function for compute the similiarity. The signature of the weight function has to be weight(i, j, e1,e2). Where e1 and e2 ara the data elements i-th and j-th respectivily, obtained via get_element, usually is a vector.\nnvec::Integer. The number of eigenvector to obtain.\nthreaded::Bool. Default: True. Specifies whether the threaded version is used.\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.clusterize-Tuple{SpectralClustering.LandmarkBasedRepresentation,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.clusterize",
    "category": "method",
    "text": "clusterize(cfg::LandmarkBasedRepresentation,X)\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.embedding-Tuple{SpectralClustering.DNCuts,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(d::DNCuts, L)\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.embedding-Tuple{SpectralClustering.NystromMethod,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::NystromMethod, X)\n\nThis is an overloaded function\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.embedding-Tuple{SpectralClustering.NystromMethod,Array{#s11,1} where #s11<:Integer,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::NystromMethod, landmarks::Vector{Int}, X)\n\nArguments\n\ncfg::[NystromMethod](@ref)\nlandmarks::Vector{Int}\nx::Any\n\nReturn values\n\n(E, L): The approximated eigenvectors, the aprooximated eigenvalues\n\nPerforms the eigenvector embedding according to\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.embedding-Tuple{SpectralClustering.NystromMethod,Array{T,2} where T,Array{T,2} where T,Array{#s10,1} where #s10<:Integer}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::NystromMethod, A::Matrix, B::Matrix, landmarks::Vector{Int})\n\nPerforms the eigenvector approximation given the two submatrices A and B.\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.create_A_B-Tuple{SpectralClustering.NystromMethod,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.create_A_B",
    "category": "method",
    "text": "create_A_B(cfg::NystromMethod, X)\n\nArguments:\n\ncfg::NystromMethod\nX\n\n#Return values\n\nSub-matrix A\nSub-matrix B\nVector{Int}. The sampled points used build the sub-matrices\n\nThis is an overloaded method. Computes the submatrix A and B according to create_A_B(::NystromMethod, ::Vector{Int}, ::Any). Returns the two submatrices and the sampled points used to calcluate it\n\n\n\n"
},

{
    "location": "man/approximate.html#SpectralClustering.create_A_B-Tuple{SpectralClustering.NystromMethod,Array{#s8,1} where #s8<:Integer,Any}",
    "page": "Approximate Embedding",
    "title": "SpectralClustering.create_A_B",
    "category": "method",
    "text": "create_A_B(cfg::NystromMethod, landmarks::Vector{Int},X)\n\nArguments:\n\ncfg::NystromMethod. The method configuration.\nlandmarks::Vector{T}. A vector of integer that containts the n indexes sampled from the data.\nX is the data that containt $ N $ patterns.\n\nLet $ W \\in \\mathbb{R}^{N \\times N}, W = \\begin{bmatrix} A & B^T \\ B & C \\end{bmatrix}, A \\in \\mathbb{R}^{ n \\times n }, B \\in \\mathbb{R}^{(N-n) \\times n}, C \\in \\mathbb{R}^{(N-n)\\times (N-n)} $ . A represents the subblock of weights among the random samples, B contains the weights from the random samples to the rest of the pixels, and C contains the weights between all of the remaining pixels. The function computes A and B from the data X using the weight function defined in cfg.\n\n\n\n"
},

{
    "location": "man/approximate.html#Content-1",
    "page": "Approximate Embedding",
    "title": "Content",
    "category": "section",
    "text": "\nPages=[\"ApproximateEmbedding.jl\"]\nModules=[SpectralClustering]"
},

{
    "location": "man/clusterize.html#",
    "page": "Eigenvector Clustering",
    "title": "Eigenvector Clustering",
    "category": "page",
    "text": ""
},

{
    "location": "man/clusterize.html#Eigenvector-Clustering-1",
    "page": "Eigenvector Clustering",
    "title": "Eigenvector Clustering",
    "category": "section",
    "text": "Once the eigenvectors are obtained, we have a continuous solution for a discrete problem. In order to obtain an assigment for every pattern,  it is needed to discretize the eigenvectors. Obtaining this discrete solution from eigenvectors often requires solving another clustering problem, albeit in a lower-dimensional space. That is, eigenvectors are treated as geometrical coordinates of a point set.This library provides two methods two obtain the discrete solution:Kmeans by means of Clustering.jl\nThe one proposed in Multiclass spectral clustering"
},

{
    "location": "man/clusterize.html#Examples-1",
    "page": "Eigenvector Clustering",
    "title": "Examples",
    "category": "section",
    "text": "Eigenvector clusterization examples"
},

{
    "location": "man/clusterize.html#Reference-Index-1",
    "page": "Eigenvector Clustering",
    "title": "Reference Index",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"man/data_access.md\"]"
},

{
    "location": "man/clusterize.html#SpectralClustering.KMeansClusterizer",
    "page": "Eigenvector Clustering",
    "title": "SpectralClustering.KMeansClusterizer",
    "category": "type",
    "text": "struct KMeansClusterizer <: EigenvectorClusterizer\n    k::Integer\n    init::Symbol\nend\n\n\n\n"
},

{
    "location": "man/clusterize.html#SpectralClustering.YuEigenvectorRotation",
    "page": "Eigenvector Clustering",
    "title": "SpectralClustering.YuEigenvectorRotation",
    "category": "type",
    "text": "Multiclass Spectral Clustering\n\n\n\n"
},

{
    "location": "man/clusterize.html#SpectralClustering.clusterize-Union{Tuple{C}, Tuple{T,C,Any}, Tuple{T}} where T<:SpectralClustering.EigenvectorEmbedder where C<:SpectralClustering.EigenvectorClusterizer",
    "page": "Eigenvector Clustering",
    "title": "SpectralClustering.clusterize",
    "category": "method",
    "text": "function clusterize{T<:EigenvectorEmbedder, C<:EigenvectorClusterizer}(cfg::T, clus::C, X)\n\nGiven a set of patterns X generates an eigenvector space according to T<:EigenvectorEmbeddder and then clusterize the eigenvectors using the algorithm defined by C<:EigenvectorClusterize.\n\n\n\n"
},

{
    "location": "man/clusterize.html#Members-Documentation-1",
    "page": "Eigenvector Clustering",
    "title": "Members Documentation",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"EigenvectorClustering.jl\"]"
},

{
    "location": "man/clusterize.html#Bibliography-1",
    "page": "Eigenvector Clustering",
    "title": "Bibliography",
    "category": "section",
    "text": "import Documenter.Documents.RawHTML\nusing DocUtils\nRawHTML(bibliography([\"stella2003multiclass\"]))"
},

{
    "location": "man/multiview.html#",
    "page": "Co-Regularized",
    "title": "Co-Regularized",
    "category": "page",
    "text": ""
},

{
    "location": "man/multiview.html#MultiView-Embedding-1",
    "page": "Co-Regularized",
    "title": "MultiView Embedding",
    "category": "section",
    "text": "When the dataset has more than one representation, each of them is named view. In the context of spectral clustering,  co-regularization techniques attempt to encourage the similarity of the examples in the new representation generated  from the eigenvectors of each view."
},

{
    "location": "man/multiview.html#Examples-1",
    "page": "Co-Regularized",
    "title": "Examples",
    "category": "section",
    "text": "MultiView Embedding examples"
},

{
    "location": "man/multiview.html#Reference-Index-1",
    "page": "Co-Regularized",
    "title": "Reference Index",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"man/data_access.md\"]"
},

{
    "location": "man/multiview.html#SpectralClustering.CoRegularizedMultiView",
    "page": "Co-Regularized",
    "title": "SpectralClustering.CoRegularizedMultiView",
    "category": "type",
    "text": "Co-regularized Multi-view Spectral Clustering\n\nAbhishek Kumar, Piyush Rai, Hal Daumé\n\n\n\n"
},

{
    "location": "man/multiview.html#SpectralClustering.View",
    "page": "Co-Regularized",
    "title": "SpectralClustering.View",
    "category": "type",
    "text": "A view\n\ntype View\n  ng_laplacian::NgLaplacian\n  lambda::Float64\nend\n\nThe type View represents the member graph is a function that returns and embedding from the data. The member lambda is a parameter that scale the eigenvectors The member nev is the number of eigenvectors requested to the embedding\n\n\n\n"
},

{
    "location": "man/multiview.html#SpectralClustering.embedding-Tuple{SpectralClustering.CoRegularizedMultiView,Array{T,1} where T}",
    "page": "Co-Regularized",
    "title": "SpectralClustering.embedding",
    "category": "method",
    "text": "embedding(cfg::CoRegularizedMultiView, X::Vector)\n\nAn example that shows how to use this methods is provied in the Usage section of the manual\n\n\n\n"
},

{
    "location": "man/multiview.html#SpectralClustering.LargeScaleMultiView",
    "page": "Co-Regularized",
    "title": "SpectralClustering.LargeScaleMultiView",
    "category": "type",
    "text": "type LargeScaleMultiView\n\nLarge-Scale Multi-View Spectral Clustering via Bipartite Graph. In AAAI (pp. 2750-2756).\n\nLi, Y., Nie, F., Huang, H., & Huang, J. (2015, January).\n\nMatlab implementation\n\nMembers\n\nk::Integer. Number of clusters.\nn_salient_points::Integer. Number of salient points.\nk_nn::Integer. k nearest neighbors.\n\'gamma::Float64`.\n\n\n\n"
},

{
    "location": "man/multiview.html#Members-Documentation-1",
    "page": "Co-Regularized",
    "title": "Members Documentation",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"MultiView.jl\"]"
},

{
    "location": "man/incremental.html#",
    "page": "Incremental",
    "title": "Incremental",
    "category": "page",
    "text": ""
},

{
    "location": "man/incremental.html#Incremental-Spectral-Clustering-1",
    "page": "Incremental",
    "title": "Incremental Spectral Clustering",
    "category": "section",
    "text": ""
},

{
    "location": "man/data_access.html#",
    "page": "Data Access",
    "title": "Data Access",
    "category": "page",
    "text": ""
},

{
    "location": "man/data_access.html#Data-Access-1",
    "page": "Data Access",
    "title": "Data Access",
    "category": "section",
    "text": "In order to establish how the data is going to be accessed, the module DataAccess provides an unified interface to access to the data for the underlaying algorithms. Every DataAccessor must implement this two methods:get_element(d::T, X, i::Integer). This function must return the i-th pattern of X.\nnumber_of_patterns(d::T,X). This function must return the numer of patterns of X"
},

{
    "location": "man/data_access.html#Reference-Index-1",
    "page": "Data Access",
    "title": "Reference Index",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"man/data_access.md\"]"
},

{
    "location": "man/data_access.html#SpectralClustering.assign!-Union{Tuple{C}, Tuple{T,C}, Tuple{T}} where T<:AbstractArray where C<:ColorTypes.Colorant",
    "page": "Data Access",
    "title": "SpectralClustering.assign!",
    "category": "method",
    "text": "function assign!(vec::T, val::C) where T<:AbstractArray where C<:Colorant\n\nThis function assigns the components of the color component val to a vector v\n\n\n\n"
},

{
    "location": "man/data_access.html#SpectralClustering.get_element!-Union{Tuple{C}, Tuple{D,Array{C,2},Array{#s5,1} where #s5<:Integer}, Tuple{D}} where D<:AbstractArray where C<:ColorTypes.Colorant",
    "page": "Data Access",
    "title": "SpectralClustering.get_element!",
    "category": "method",
    "text": "function get_element!(o::Matrix,  img::Matrix{C}, i::Vector{Integer}) where C<:Colorant\n\n\n\n\n"
},

{
    "location": "man/data_access.html#SpectralClustering.get_element!-Union{Tuple{C}, Tuple{T,Array{C,2},Integer}, Tuple{T}} where T<:AbstractArray where C<:ColorTypes.Colorant",
    "page": "Data Access",
    "title": "SpectralClustering.get_element!",
    "category": "method",
    "text": "get_element!{T<:AbstractArray}(vec::T,  img::Matrix{Gray}, i::Integer)\n\nReturn throughvec the intensity image element  [x,y, i], where xy are the spatial position of the pixel and the value i of the pixel (xy).\n\n\n\n"
},

{
    "location": "man/data_access.html#SpectralClustering.get_element-Union{Tuple{Array{T,2},Array{T,1} where T}, Tuple{T}} where T<:ColorTypes.Colorant",
    "page": "Data Access",
    "title": "SpectralClustering.get_element",
    "category": "method",
    "text": "function get_element( img::Matrix{RGB}, i::Vector) \n\n\n\n"
},

{
    "location": "man/data_access.html#SpectralClustering.number_of_patterns-Union{Tuple{Array{T,2}}, Tuple{T}} where T<:ColorTypes.Colorant",
    "page": "Data Access",
    "title": "SpectralClustering.number_of_patterns",
    "category": "method",
    "text": "number_of_patterns{T<:Any}(X::Array{T,3})\n\nReturn the number of pixels in the image\n\n\n\n"
},

{
    "location": "man/data_access.html#SpectralClustering.spatial_position-Tuple{Array{T,2} where T,Int64}",
    "page": "Data Access",
    "title": "SpectralClustering.spatial_position",
    "category": "method",
    "text": " spatial_position(X::Matrix, i::Int)\n\nReturns the sub indexes from the linear index i\n\n\n\n"
},

{
    "location": "man/data_access.html#Members-Documentation-1",
    "page": "Data Access",
    "title": "Members Documentation",
    "category": "section",
    "text": "Modules = [SpectralClustering]\nPages=[\"Utils/DataAccess.jl\"]"
},

{
    "location": "man/graph.html#",
    "page": "Graph",
    "title": "Graph",
    "category": "page",
    "text": ""
},

{
    "location": "man/graph.html#Graphs-1",
    "page": "Graph",
    "title": "Graphs",
    "category": "section",
    "text": ""
},

{
    "location": "man/graph.html#Reference-1",
    "page": "Graph",
    "title": "Reference",
    "category": "section",
    "text": ""
},

{
    "location": "man/graph.html#Index-1",
    "page": "Graph",
    "title": "Index",
    "category": "section",
    "text": "Pages=[\"man/graph.md\"]\nModules=[SpectralClustering]"
},

{
    "location": "man/graph.html#SpectralClustering.Graph",
    "page": "Graph",
    "title": "SpectralClustering.Graph",
    "category": "type",
    "text": "Graph(n_vertices::Integer=0; vertex_type::DataType  = Any ,initial_value=nothing, weight_type::DataType = Float64)\n\nConstruct an undirected weighted grpah of n_vertices vertices.\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.connect!-Tuple{SpectralClustering.Graph,Integer,Array{T,1} where T,Array{T,1} where T}",
    "page": "Graph",
    "title": "SpectralClustering.connect!",
    "category": "method",
    "text": "function connect!(g::Graph, i::Integer, neighbors::Vector, weigths::Vector)\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.connect!-Tuple{SpectralClustering.Graph,Integer,Integer,Number}",
    "page": "Graph",
    "title": "SpectralClustering.connect!",
    "category": "method",
    "text": "connect!(g::Graph,i::Integer,j::Integer,w::Number)\n\nConnect the vertex i with the vertex j with weight w.\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.disconnect-Tuple{SpectralClustering.Graph,Integer,Integer}",
    "page": "Graph",
    "title": "SpectralClustering.disconnect",
    "category": "method",
    "text": "disconnect(g::Graph,i::Integer,j::Integer)\n\nRemoves the edge that connects the i-th vertex to the j-th vertex.\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.number_of_vertices-Tuple{SpectralClustering.Graph}",
    "page": "Graph",
    "title": "SpectralClustering.number_of_vertices",
    "category": "method",
    "text": "number_of_vertices(g::Graph)\n\nReturn the number of vertices of g.\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.random_graph-Tuple{Integer}",
    "page": "Graph",
    "title": "SpectralClustering.random_graph",
    "category": "method",
    "text": "function random_graph(iterations::Integer; probs=[0.4,0.4,0.2], weight=()->5, debug=false)\n\nCreate a random graphs. probs is an array of probabilities. The function create a vertex with probability probs[1], connect two vertices with probability probs[2] and delete a vertex with probability probs[2]. The weight of the edges is given by weight\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.remove_vertex!-Tuple{SpectralClustering.Graph,Integer}",
    "page": "Graph",
    "title": "SpectralClustering.remove_vertex!",
    "category": "method",
    "text": "remove_vertex!(g::Graph,i::Integer)\n\nRemove the i-th vertex.\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.target_vertex-Tuple{SpectralClustering.Edge,SpectralClustering.Vertex}",
    "page": "Graph",
    "title": "SpectralClustering.target_vertex",
    "category": "method",
    "text": "target_vertex(e::Edge,v::Vertex)\n\nGiven an edge e and a vertex v returns the other vertex different from v\n\n\n\n"
},

{
    "location": "man/graph.html#SpectralClustering.Edge",
    "page": "Graph",
    "title": "SpectralClustering.Edge",
    "category": "type",
    "text": "type Edge\n\n\n\n"
},

{
    "location": "man/graph.html#Base.length-Tuple{SpectralClustering.Vertex}",
    "page": "Graph",
    "title": "Base.length",
    "category": "method",
    "text": "length(v::Vertex)\n\nReturn the number of edges connected to a given vertex.\n\n\n\n"
},

{
    "location": "man/graph.html#Content-1",
    "page": "Graph",
    "title": "Content",
    "category": "section",
    "text": "\nPages=[\"Graph/Graphs.jl\"]\nModules=[SpectralClustering]"
},

{
    "location": "man/landmark_selection.html#",
    "page": "Landmarks Selection",
    "title": "Landmarks Selection",
    "category": "page",
    "text": ""
},

{
    "location": "man/landmark_selection.html#Landmark-Selection-1",
    "page": "Landmarks Selection",
    "title": "Landmark Selection",
    "category": "section",
    "text": "In order to avoid the construction of a complete similarity matrix some spectral clustering methods compute the simmilarity function between a subset of patterns. This module provides an interface to sample points from diferentes data structures.Methods availaible:Random . This selection method samples k random points from a dataset\nEvenlySpaced. This selection method samples spaced evenly acorrding ther index."
},

{
    "location": "man/landmark_selection.html#Detailed-Description-1",
    "page": "Landmarks Selection",
    "title": "Detailed Description",
    "category": "section",
    "text": ""
},

{
    "location": "man/landmark_selection.html#Random-Landmark-Selection-1",
    "page": "Landmarks Selection",
    "title": "Random Landmark Selection",
    "category": "section",
    "text": "using SpectralClustering\nnumber_of_points    = 20\ndimension           = 5\ndata                = rand(dimension,number_of_points)\nselector            = RandomLandmarkSelection()\nnumber_of_landmarks = 7\nselect_landmarks(selector, number_of_landmarks, data )"
},

{
    "location": "man/landmark_selection.html#Evenly-Spaced-Landmark-Selection-1",
    "page": "Landmarks Selection",
    "title": "Evenly Spaced Landmark Selection",
    "category": "section",
    "text": "using SpectralClustering\nnumber_of_points    = 20\ndimension           = 5\ndata                = rand(dimension,number_of_points)\nselector            = EvenlySpacedLandmarkSelection()\nnumber_of_landmarks = 5\nselect_landmarks(selector, number_of_landmarks, data )\n"
},

{
    "location": "man/landmark_selection.html#Index-1",
    "page": "Landmarks Selection",
    "title": "Index",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"man/landmark_selection.md\"]"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.AbstractLandmarkSelection",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.AbstractLandmarkSelection",
    "category": "type",
    "text": "abstract type AbstractLandmarkSelection end\n\nAbstract type that defines how to sample data points. Types that inherint from AbstractLandmarkSelection has to implements the following interface:\n\nselect_landmarks{L<:AbstractLandmarkSelection}(c::L, X)\n\nThe select_landmarksfunction returns an array with the indices of the sampled points.\n\nArguments\n\nc::T<:AbstractLandmarkSelecion. The landmark selection type.\nd::D<:DataAccessor.  The DataAccessor type.\nX. The data. The data to be sampled.\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.EvenlySpacedLandmarkSelection",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.EvenlySpacedLandmarkSelection",
    "category": "type",
    "text": "type EvenlySpacedLandmarkSelection <: AbstractLandmarkSelection\n\nThe EvenlySpacedLandmarkSelection selection method selects  n evenly spaced points  from a dataset.\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.MS3",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.MS3",
    "category": "type",
    "text": "type MS3 <: AbstractLandmarkSelection\n    proportion::Float64\n    sim::Function\nend\n\nThe MS3 selection method selects  m  NYSTROM SAMPLING DEPENDS ON THE EIGENSPECTRUM SHAPE OF THE DATA\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.RandomLandmarkSelection",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.RandomLandmarkSelection",
    "category": "type",
    "text": "type RandomLandmarkSelection <: LandmarkSelection\n\nRandom random samples n data points from a dataset.\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.select_landmarks-Tuple{SpectralClustering.EvenlySpacedLandmarkSelection,Integer,Any}",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.select_landmarks",
    "category": "method",
    "text": "select_landmarks(c::EvenlySpacedLandmarkSelection,n::Integer, X)\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#SpectralClustering.select_landmarks-Tuple{SpectralClustering.RandomLandmarkSelection,Integer,Any}",
    "page": "Landmarks Selection",
    "title": "SpectralClustering.select_landmarks",
    "category": "method",
    "text": "select_landmarks(c::RandomLandmarkSelection,d::T,n::Integer, X)\n\nThe function returns nrandom points according to RandomLandmarkSelection\n\nArguments\n\nc::RandomLandmarkSelection.\nn::Integer. The number of data points to sample.\nX. The data to be sampled.\n\n\n\n"
},

{
    "location": "man/landmark_selection.html#Content-1",
    "page": "Landmarks Selection",
    "title": "Content",
    "category": "section",
    "text": "Modules=[SpectralClustering]\nPages=[\"src/LandmarkSelection.jl\"]"
},

{
    "location": "man/graphmatrices.html#",
    "page": "Graph Matrices",
    "title": "Graph Matrices",
    "category": "page",
    "text": ""
},

{
    "location": "man/graphmatrices.html#Graph-matrices-1",
    "page": "Graph Matrices",
    "title": "Graph matrices",
    "category": "section",
    "text": "The Graphs.Matrices  module provides a collecion of functions that computes differents matrices  from a graph. Given a Graph G=(VE) where (v_iv_jw_ij) in E v_iv_j in V w_ij in R this module allows to obtain:The simmiliarty matrix W. Where W_ij = w_ij\nThe laplacian matrix L=(D-W). Where D = W mathbf1\nThe normalized laplacian L_N = D^-frac12 (D-W) D^-frac12\nNSJ Laplacian  L_NSJ = D^-1W"
},

{
    "location": "man/graphmatrices.html#Detailed-Description-1",
    "page": "Graph Matrices",
    "title": "Detailed Description",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphmatrices.html#Laplacian-1",
    "page": "Graph Matrices",
    "title": "Laplacian",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphmatrices.html#Normalized-Laplacian-1",
    "page": "Graph Matrices",
    "title": "Normalized Laplacian",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphmatrices.html#NSJ-Laplacian-1",
    "page": "Graph Matrices",
    "title": "NSJ Laplacian",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphmatrices.html#Reference-1",
    "page": "Graph Matrices",
    "title": "Reference",
    "category": "section",
    "text": ""
},

{
    "location": "man/graphmatrices.html#Index-1",
    "page": "Graph Matrices",
    "title": "Index",
    "category": "section",
    "text": "Pages=[\"Graph/Matrices.jl\"]\nModules=[SpectralClustering]"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.degrees-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.degrees",
    "category": "method",
    "text": "degrees(g::Graph)\n\nReturns a vector with the degrees of every vertex\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.incidence-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.incidence",
    "category": "method",
    "text": "incidence(g::Graph)\n\nReturns the incidence matrix of the graph ´´´G´´´\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.laplacian-Tuple{SparseMatrixCSC,Array{T,1} where T}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.laplacian",
    "category": "method",
    "text": "laplacian(W::SparseMatrixCSC, d::Vector)\n\nReturns the laplacian matrix ´´´L´´´ of the graph ´´´g´´´\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.laplacian-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.laplacian",
    "category": "method",
    "text": "laplacian(g::Graph)\n\nReturns the laplacian matrix ´´´L´´´ of the graph ´´´g´´´\n\nL=D-W\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.ng_laplacian-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.ng_laplacian",
    "category": "method",
    "text": "ng_laplacian(g::Graph)\n\nReturns the laplacian matrix of the graph ´´´G´´´ according to \"\"On Spectral Clustering: Analysis and an algorithm\"\" of Andrew Y. Ng, Michael I. Jordan and Yair Weiss.\n\nL = D^-frac12 W D^-frac12\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.normalized_laplacian-Tuple{SparseMatrixCSC,Any}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.normalized_laplacian",
    "category": "method",
    "text": "normalized_laplacian(W::SparseMatrixCSC,d)\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.normalized_laplacian-Tuple{SparseMatrixCSC}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.normalized_laplacian",
    "category": "method",
    "text": " normalized_laplacian(W::SparseMatrixCSC)\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.normalized_laplacian-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.normalized_laplacian",
    "category": "method",
    "text": "normalized_laplacian(g::Graph)\n\nIf W it is the weight matrix W of ´´´g´´´, D=W mathbf1. The normalized laplacian L = D^-frac12 (D-W) D^-frac12. The function returns the normalized laplacian of the graph ´´´g´´´ and D^frac12.\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.weight_matrix-Tuple{SpectralClustering.Graph,Array{Int64,1},Array{Int64,1}}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.weight_matrix",
    "category": "method",
    "text": "weight_matrix(g::Graph)\n\nGiven a graph ´´´g´´´ and two array of vertices (set1, set2) indexes returns the weight matrix form the vertices in set1 to the vertices in set2.\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.weight_matrix-Tuple{SpectralClustering.Graph}",
    "page": "Graph Matrices",
    "title": "SpectralClustering.weight_matrix",
    "category": "method",
    "text": "weight_matrix(g::Graph)\n\nGiven a graph ´g´ returns the weight matrix and a vector with the vertices degrees.\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#SpectralClustering.weight_matrix-Union{Tuple{T}, Tuple{Type{T},SpectralClustering.Graph}} where T<:AbstractFloat",
    "page": "Graph Matrices",
    "title": "SpectralClustering.weight_matrix",
    "category": "method",
    "text": "julia weight_matrix{T<:AbstractFloat}(n_type::T,g::Graph)`\n\nGiven a graph ´g´ returns the weight matrix of `n_type and a vector with the vertices degrees.\n\n\n\n"
},

{
    "location": "man/graphmatrices.html#Content-1",
    "page": "Graph Matrices",
    "title": "Content",
    "category": "section",
    "text": "\nPages=[\"Graph/Matrices.jl\"]\nModules=[SpectralClustering]"
},

]}
