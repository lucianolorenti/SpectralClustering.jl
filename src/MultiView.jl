export embedding,
       View,
    CoRegularizedMultiView,
    KernelProduct,
    KernelAddition

"""
A view

```julia
type View
  ng_laplacian::NgLaplacian
  lambda::Float64
end
```
The type View represents
the member graph is a function that returns and embedding from the data.
The member lambda is a parameter that scale the eigenvectors
The member nev is the number of eigenvectors requested to the embedding
"""
type View
    embedder::EigenvectorEmbedder    
    lambda::Float64
end

"""
# Co-regularized Multi-view Spectral Clustering

### Abhishek Kumar, Piyush Rai, Hal Daumé

"""
type CoRegularizedMultiView <: EigenvectorEmbedder
    threshold::Float64
    views::Vector{View}
end
"""
```julia
embedding(cfg::CoRegularizedMultiView, X::Vector)
```

An example that shows how to use this methods is provied in the Usage section of the manual
"""
function embedding(cfg::CoRegularizedMultiView, X::Vector; disagreement::Union{Void,Vector} = nothing)
   U = Vector{Matrix}(length(cfg.views))
   Laplacians = [ ng_laplacian(X[i])[1] for i=1:length(cfg.views) ]
    #Initialize all U(v),2≤v≤m$
    for i=2:length(cfg.views)
        U[i] =  embedding(cfg.views[i].embedder,Laplacians[i])
    end
   curr_objective = -Inf
   prev_objective = 0
   best_objective = Inf
   iterations_without_improvement = 0
    while (abs(curr_objective - prev_objective)  > cfg.threshold) && (iterations_without_improvement < 5)
        for i=1:length(cfg.views)
           L = Laplacians[i]
            for j=1:length(cfg.views)
                if (j!=i)
                    L = L + cfg.views[i].lambda*U[j]*U[j]'
                end
            end
            U[i] = embedding(cfg.views[i].embedder, L)
            prev_objective = curr_objective
            curr_objective = sum([trace((U[i]*U[i]')*(U[j]*U[j]')) for j=1:length(U)  for d=1:length(U)])
            if curr_objective < best_objective
                best_objective = curr_objective
                iterations_without_improvement = 0
            else
                iterations_without_improvement=iterations_without_improvement+1
            end
            if (disagreement!=nothing)
                push!( disagreement, curr_objective)
            end
            
        end
    end
    return U[1]
end

type KernelAddition <: EigenvectorEmbedder
    embedder::EigenvectorEmbedder    
end
function embedding(cfg::KernelAddition, X::Vector)
    (W,_) = weight_matrix(X[1])
    for j=2:length(X)
        (W_1,_) = weight_matrix(X[j])
        W = W + W_1
    end
    return  embedding(cfg.embedder, W)
end
type KernelProduct <: EigenvectorEmbedder
    embedder::EigenvectorEmbedder
end
function embedding(cfg::KernelProduct, X::Vector)
    (W,_) = weight_matrix(X[1])
    for j=2:length(X)
        (W_1,_) = weight_matrix(X[j])
        W = W .* W_1
    end
    return embedding(cfg.embedder, W)
end

"""
type LargeScaleMultiView
# Large-Scale Multi-View Spectral Clustering via Bipartite Graph. In AAAI (pp. 2750-2756).
## Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January).

[Matlab implementation](https://github.com/zzz123xyz/MVSC/blob/master/MVSC.m)
# Members
- `k::Integer`. Number of clusters.
- `n_salient_points::Integer`. Number of salient points.
- `k_nn::Integer`. k nearest neighbors.
- 'gamma::Float64`.
"""
type LargeScaleMultiView
	k::Integer
	n_salient_points::Integer
	k_nn::Integer
	gamma::Float64
end
"""
# Parameters
- `cfg::LargeScaleMultiView`
- `data::Vector`. An array of views.
"""

#=function embedding(cfg::LargeScaleMultiView, data::Vector)
	niters = 10;
    n =
	V = length(data)
	salient_points =
    rest_of_points =

	a = fill(1/nbclusters, [V]);
	for v = 1:V
	    RestPnt = data{v}(:,RestPntInd)';
    PairDist = pdist2(RestPnt,SltPnt(:,dim_V_ind1(v): dim_V_ind2(v)));
    [score, ind] = sort(PairDist,2);
    ind = ind(:,1:k);

%*****
%make a Indicator Mask to record j \in \theta_i
    IndMask = zeros(n - nbSltPnt, nbSltPnt);
    for i = 1:n - nbSltPnt
         IndMask(i, ind(i,:)) = 1;
    end

    Kernel = exp(-(PairDist).^2 ./ (2*param^2));
    Kernel = Kernel.*IndMask;

    SumSltKnl = repmat(sum(Kernel, 2),[1,nbSltPnt]);
    Z{v} = Kernel ./ SumSltKnl;
    Dc{v} = diag(sum(Z{v},1)+eps);
    Dr{v} = diag(sum(Z{v},2));
    D{v} = blkdiag(Dr{v},Dc{v});

    tmp1 = zeros(n);
    tmp1(1:n-nbSltPnt,n-nbSltPnt+1:end) = Z{v};
    tmp1(n-nbSltPnt+1:end,1:n-nbSltPnt) = Z{v}';
    W{v} = tmp1;

    L{v} = eye(n) - (D{v}^-0.5) * W{v} * (D{v}^-0.5);
end

for t = 1:niters
    L_sum = zeros(n, n);
    Z_hat = zeros(n - nbSltPnt, nbSltPnt);

    for v = 1:V
        Z_hat = Z_hat + a(v)^gamma*Z{v}*(Dc{v})^(-0.5);
    end

    % compute G according to (14)
    [Gx_a, S, Gu_a] = svd(Z_hat, 'econ');
    Gx = Gx_a(:,1:nbclusters);
    Gu = Gu_a(:,1:nbclusters);
    G = [Gx', Gu']';

    for v = 1:V
        h(v) = trace(G'*L{v}*G); %*** h(v) = trace(G'*L{v}*G); dim of G mismatch L
    end

    % compute a(v) according to (10)
    tmp1 = (gamma .* h).^(1/(1-gamma)) ;
    a = tmp1 ./ sum(tmp1);

    [Y, C] = kmeans(G, nbclusters);

    % compute the value of objective function (5)
    for v = 1:V
        L_sum = L_sum + a(v)^gamma*L{v};
    end
    obj_value(t) = trace(G'*L_sum*G);  %obj_value(t) = trace(G'*L_sum*G);

end
=#
