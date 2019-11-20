using SpectralClustering
using Distances
using Test
using LinearAlgebra
using Statistics
using Clustering
using Images
using Random
import LightGraphs.LinAlg: adjacency_matrix
number_of_vertices = 5
Random.seed!(0)


function two_gaussians(N::Integer = 500; std_1=5, std_2 = 5, center_1=[15,5], center_2=[-15, 5])
    d1 = (randn(2, N) * std_1) .+ center_1
    d2 = (randn(2, N) * std_2) .+ center_2
    labels = round.(Integer, vcat(zeros(N), ones(N)))
    return (hcat(d1, d2), labels)
end

function three_gaussians(N::Integer = 250; )
    d1 = (randn(2, N) * 1.5) .+ [5, 0]
    d2 = (randn(2, N) * 1) .+ [0, 0]
    d3 = (randn(2, N) * 1.5) .+ [-5, 0]
    labels = round.(Integer, vcat(zeros(N), ones(N), ones(N)* 2))
    return (hcat(d1, d2, d3), labels)
end


@testset "Graph Creation" begin
    @testset "KNNNeighborhood" begin
        function weight(i::Integer, neigh, v, m)
            return Distances.colwise(SqEuclidean(), m, v)
        end
        data = convert.(Float64, [2 2; -2 -2; 2 -2; -2 2;0  0]')
        knnconfig = KNNNeighborhood(data, 3);
        graph = create(knnconfig, weight, data);
        v1 = graph.vertices[1]
        @test v1.number_of_edges == 3
        for edge in v1
            v2 = target_vertex(edge, v1)
            @test norm(data[:, v2.id] - [-2, -2]) > 0.001
        end
        @test graph.vertices[2].connections == Set([4,3,5])
        @test graph.vertices[end].number_of_edges == 4
    end
    @testset "RandomNeighborhood" begin
        data, labels = two_gaussians()
        randomconfig = RandomNeighborhood(5)
        graph = create(randomconfig, ones, data);
        @test all([v.number_of_edges for v in graph.vertices] .>= 5)
    end
    @testset "CliqueNeighborhood" begin
        data, labels = two_gaussians(15)
        clique = CliqueNeighborhood()
        graph = create(clique, ones, data);
        @test all([v.number_of_edges for v in graph.vertices] .== 29)
        @test all([v.degree for v in graph.vertices] .== 29)
    end
    @testset "PixelNeighborhood" begin
        img = fill(RGB(1, 0, 0), 20, 20)
        nconfig = PixelNeighborhood(1)
        graph = create(nconfig, ones, img)
        @test length(graph.vertices) == size(img, 1) * size(img, 2)
        @test graph.vertices[1].number_of_edges == 3
        @test graph.vertices[148].number_of_edges == 8
    end
    @testset "Local Sale Knn" begin
        function weight(i::Integer, neigh, v, m)
            return Distances.colwise(Euclidean(), m, v)
        end
        X = [5.0 5; 5 4; 4 4; 6 6; -10 -10; -9 -9; -8 -8; -11 -11]'
        knnconfig = KNNNeighborhood(X, 3)
        scale = local_scale(knnconfig, weight, X, k = 3)
        @test isapprox(scale[1], sqrt(2))
        @test isapprox(scale[end], sqrt(18))
    end
    @testset "Local Sale Image" begin
        function weight(i::Integer, neigh, v, m)
            col_dist = Distances.colwise(Euclidean(), m[2:end, :], v[2:end])
            xy_dist = Distances.colwise(Euclidean(), m[1:2, :], v[1:2])
            return hcat(col_dist, xy_dist)
        end
        X = rand(RGB, 50, 50)
        knnconfig = PixelNeighborhood(3)
        scale = local_scale(knnconfig, weight, X, k = 9)
        @test size(scale) == (2, 50 * 50)
    end
end;

@testset "Embedding" begin
    @testset "NgLaplacian" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        (data, labels) = two_gaussians()


        knnconfig = KNNNeighborhood(data, 7)
        graph = create(knnconfig, weight, data)
        emb = embedding(NgLaplacian(1), graph)
        pred_clustering = convert(Array{Int64}, (emb .<= mean(emb)))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
    @testset "ShiMalikLaplacian" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        (data, labels) = two_gaussians()
        knnconfig = KNNNeighborhood(data, 15)
        graph = create(knnconfig, weight, data)
        emb = embedding(ShiMalikLaplacian(1), graph)
        pred_clustering = convert(Array{Int64}, (emb .<= mean(emb)))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
    @testset "PartialGroupingConstraints" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 0.7)
        end
        N = 150
        (d, labels) = three_gaussians(N)
        knnconfig = KNNNeighborhood(d, 100)
        graph = create(knnconfig, weight, d)

        indices_clus_1 = [1, 2, 3, 4, 5]
        indices_clus_2 = [N+1, N+2, N+3, N+4]
        indices_clus_3 = [2*N+1, 2*N+2, 2*N+3, 2*N+4]

        constraints = Vector{Integer}[ vcat(indices_clus_1, indices_clus_2) ] ;
        emb_1 = embedding(PartialGroupingConstraints(1, smooth=true),  graph, constraints)
        labels_1 = vcat(zeros(Integer, N*2), ones(Integer, N))

        constraints = Vector{Integer}[ vcat(indices_clus_2, indices_clus_3) ] 
        emb_2 = embedding(PartialGroupingConstraints(1, smooth=true),  graph, constraints)
        labels_2 = vcat(zeros(Integer, N), ones(Integer, N*2))

        pred_clustering = convert(Array{Int64}, (emb_1 .<= mean(emb_1)))
        @test randindex(pred_clustering, labels_1)[4] > 0.85
        @test randindex(pred_clustering, labels_2)[4] < 0.5

        pred_clustering = convert(Array{Int64}, (emb_2 .<= mean(emb_2)))
        @test randindex(pred_clustering, labels_2)[4] > 0.85
        @test randindex(pred_clustering, labels_1)[4] < 0.5
    end
end
@testset "Clustering" begin
    @testset "KMeans Clustering" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        (data, labels) = two_gaussians()
        knnconfig = KNNNeighborhood(data, 7)
        graph = create(knnconfig, weight, data)
        (data, labels) = two_gaussians()
        pred_clustering = clusterize(NgLaplacian(2), KMeansClusterizer(2), graph)
        @test randindex(pred_clustering.assignments, labels)[4] > 0.9
    end
    @testset "YuEigenvectorRotation" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        (data, labels) = two_gaussians()
        knnconfig = KNNNeighborhood(data, 7)
        graph = create(knnconfig, weight, data)
        (data, labels) = two_gaussians()
        pred_clustering = clusterize(NgLaplacian(2), YuEigenvectorRotation(500), graph)
        @test randindex(pred_clustering.assignments, labels)[4] > 0.9
    end

end
@testset "Landmark Selection" begin
    @testset "RandomLandmarkSelection" begin
        r = RandomLandmarkSelection()
        data = rand(2, 25)
        s = select_landmarks(r, 15, data)
        @test length(s) == 15
        @test length(unique(s)) == length(s)
        @test minimum(s)>=1 && maximum(s)<= 25
    end
    @testset "EvenlySpacedLandmarkSelection" begin
        e = EvenlySpacedLandmarkSelection()
        data = rand(2, 25)
        s = select_landmarks(e, 5, data)
        @test length(s) == 5
        @test all(diff(s) .== 5)
        @test minimum(s)>=1 && maximum(s)<= 25
    end
    @testset "BresenhamLandmarkSelection" begin
        e = BresenhamLandmarkSelection()
        data = rand(2, 25)
        s = select_landmarks(e, 5, data)
        @test length(s) == 5
        @test all(diff(s) .> 0)
        @test minimum(s)>=1 && maximum(s)<= 25
    end
end
@testset "Approximate Embedding" begin
    @testset "Nystrom" begin
        @testset "Data embedding" begin
            function weight(i::Integer, neigh, v, m)
                return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
            end
            (data, labels) = two_gaussians(6000)
            embedding_config = NystromMethod(EvenlySpacedLandmarkSelection(), 1000, weight, 1)
            emb = embedding(embedding_config, data)
            pred_clustering = convert(Array{Int64}, (emb .<= mean(emb)))
            @test randindex(pred_clustering, labels)[4] > 0.9
        end
        @testset "Image embedding" begin
            function weight(i::Integer,j::Vector{<:Integer},pixel_i, neighbors_data)
                data_diff = pixel_i[3:5] .- neighbors_data[3:5,:]
                a = exp.(-abs.(data_diff)./(2*0.1^2))
                a = prod(a, dims=1)
                return vec(a)
            end

            img = fill(RGB(0,0,0), 50, 50)

            cluster_1 = CartesianIndices(size(img))[5:20, 5:20]
            cluster_2 = CartesianIndices(size(img))[35:42, 35:47]
            img[cluster_1] .= RGB(1, 0, 0)
            img[cluster_2] .= RGB(0, 0, 1)

            labels = zeros(Integer, 50*50)
            labels[LinearIndices(size(img))[cluster_1][:]] .= 1
            labels[LinearIndices(size(img))[cluster_2][:]] .= 2

            embedding_config = NystromMethod(EvenlySpacedLandmarkSelection(),
                                            500, weight, 1)
            emb = embedding(embedding_config, img)
            emb = vec(round.(emb, digits=2))
            clusters = zeros(Integer, 50*50)
            for (i, val) in enumerate(unique(emb))
                clusters[findall(emb.==val)] .= i -1
            end
            @test clusters == labels
        end
    end
    @testset "DNCuts" begin
        function weight(i::Integer,j::Vector{<:Integer},pixel_i, neighbors_data)
            data_diff = pixel_i[3:5] .- neighbors_data[3:5,:]
            a = exp.(-((data_diff).^2)./(0.1))
            a = prod(a, dims=1)
            return vec(a)
        end
        img = fill(RGB(0,0,0), 50, 50)

        cluster_1 = CartesianIndices(size(img))[5:20, 5:20]
        cluster_2 = CartesianIndices(size(img))[35:42, 35:47]
        img[cluster_1] .= RGB(1, 0, 0)
        img[cluster_2] .= RGB(0, 0, 1)

        labels = ones(Integer, 50*50)
        labels[LinearIndices(size(img))[cluster_1][:]] .= 2
        labels[LinearIndices(size(img))[cluster_2][:]] .= 3

        nconfig = PixelNeighborhood(4)
        graph = create(nconfig, weight, img);

        dncuts = DNCuts(2, 2, size(img))
        emb = embedding(dncuts, graph)
        pred_clustering = clusterize(dncuts, KMeansClusterizer(3), graph)
        @test randindex(pred_clustering.assignments, labels)[4] > 0.9
    end
    @testset "LandmarkBasedRepresentation" begin
        function weight(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        (data, labels) = two_gaussians(6000)
        cfg = LandmarkBasedRepresentation(
                BresenhamLandmarkSelection(),
                500,
                25,
                2,
                weight,
                true)
        emb = embedding(cfg, data)
        pred_clustering = convert(Array{Int64}, (emb[:, 1] .<= mean(emb[:, 1])))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
end
@testset "MultiView" begin
    @testset "CoRegularizedMultiView" begin
        function weight_1(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        function weight_2(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 45)
        end
        (data, labels) = two_gaussians(500, center_1=[-15, -15], center_2=[9, 9])
        knnconfig = KNNNeighborhood(data, 7)
        graph_1 = create(knnconfig, weight_1, data);
        graph_2 = create(knnconfig, weight_2, data);
        coreg = CoRegularizedMultiView([View(1, 0.001),
                                        View(1, 0.001)])
        emb = embedding(coreg, [graph_1, graph_2])
        pred_clustering = convert(Array{Int64}, (emb[:, 1] .<= mean(emb[:, 1])))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
    @testset "KernelProduct" begin
        function weight_1(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        function weight_2(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 45)
        end
        (data, labels) = two_gaussians(500, center_1=[-15, -15], center_2=[9, 9])
        knnconfig = KNNNeighborhood(data, 7)
        graph_1 = create(knnconfig, weight_1, data);
        graph_2 = create(knnconfig, weight_2, data);
        kernel_addition = KernelProduct(NgLaplacian(2))
        emb = embedding(kernel_addition, [graph_1, graph_2])
        pred_clustering = convert(Array{Int64}, (emb[:, 1] .<= mean(emb[:, 1])))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
    @testset "KernelAddition" begin
        function weight_1(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 15)
        end
        function weight_2(i::Integer, neigh, v, m)
            return exp.(-Distances.colwise(SqEuclidean(), m, v) / 45)
        end
        (data, labels) = two_gaussians(500, center_1=[-15, -15], center_2=[9, 9])
        knnconfig = KNNNeighborhood(data, 7)
        graph_1 = create(knnconfig, weight_1, data);
        graph_2 = create(knnconfig, weight_2, data);
        kernel_addition = KernelAddition(NgLaplacian(2))
        emb = embedding(kernel_addition, [graph_1, graph_2])
        pred_clustering = convert(Array{Int64}, (emb[:, 1] .<= mean(emb[:, 1])))
        @test randindex(pred_clustering, labels)[4] > 0.9
    end
end

@testset "Utils" begin
    @testset "Normalization" begin
        a = rand(50, 3)
        @test isapprox(norm(SpectralClustering.normalize_rows(a)[1, :]), 1)
        @test isapprox(norm(SpectralClustering.normalize_cols(a)[:, 1]), 1)
        a = rand(50, 1)
        @test isapprox(norm(SpectralClustering.normalize_rows(a)[1, :]), a[1])
    end
end
@testset "DataAcces" begin
    a = rand(50, 3)
    @test number_of_patterns(a) == 3
    @test get_element(a, 1) == a[:, 1]
    a = a'
    @test number_of_patterns(a) == 50
    @test get_element(a, 1) == a[:, 1]

    a = rand(RGB, 50 ,50)
    @test number_of_patterns(a) == 50*50
end