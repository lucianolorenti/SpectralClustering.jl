module EigenvectorRotation
export discretize,
       DiscretizeEigenvectorsResult
#=
type Config
      mMethod::Integer
      mNumDims::Integer
      mNumData::Integer
      mNumAngles::Integer
      ik::Vector
      jk::Vector;

      mX::Matrix
      mXrot::Union{Void,Matrix}
      mQuality::Float32
      mClusters::Vector{Vector{Integer}}
end

function Config(X, method::Integer)
      (ndata,ndims) = size(X)
      num_angles = round(Integer,(ndims*(ndims-1)/2))
      ik = zeros(Integer,num_angles)
      jk = zeros(Integer,num_angles)
      #build index mapping (to index upper triangle)
      k=1
      for i=1:ndims-1
            for  j=i+1:ndims
                  ik[k] = i;
                  jk[k] = j;
                  k=k+1
            end
      end
      clusters = Vector{Vector{Integer}}(ndims)
      for i=1:ndims
            clusters[i] =Vector{Integer}()
      end
      return Config(method, ndims,ndata,num_angles,ik,jk,X,nothing,0,clusters)
end

function  evrot(e::Config)
      #definitions
      max_iter = 200;
      theta = zeros(e.mNumAngles);
      theta_new = zeros(e.mNumAngles);
      Q = evqual(e,e.mX); # initial quality
      Q_old1 = Q;
      Q_old2 = Q;
      iter = 0;
      while  iter < max_iter  # iterate to refine quality
            iter = iter + 1
            for d=1:e.mNumAngles
                  if  e.mMethod == 2 # descend through numerical drivative
                        alpha = 0.1;
                        # move up
                        theta_new[d] = theta[d] + alpha;
                        Xrot = rotate_givens(e,theta_new);
                        Q_up = evqual(e,Xrot);
                        #move down
                        theta_new[d] = theta[d] - alpha;
                        Xrot = rotate_givens(e,theta_new);
                        Q_down = evqual(e,Xrot);
                        # update only if at least one of them is better
                        if   Q_up > Q || Q_down > Q
                              if  Q_up > Q_down
                                    theta[d] = theta[d] + alpha;
                                    theta_new[d] = theta[d];
                                    Q = Q_up;
                              else
                                    theta[d] = theta[d] - alpha;
                                    theta_new[d] = theta[d];
                                    Q = Q_down;
                              end
                        end
                  else  # descend through true derivative
                        alpha = 1.0;
                        dQ = evqualitygrad(e,theta, d);
                        theta_new[d] = theta[d] - alpha * dQ;
                        Xrot = rotate_givens(e,theta_new);
                        Q_new = evqual(e,Xrot);

                        if  Q_new > Q
                              theta[d] = theta_new[d];
                              Q = Q_new;
                        else
                              theta_new[d] = theta[d];
                        end
                  end
            end
            #stopping criteria
            if  iter > 2
                  if( Q - Q_old2 < 0.001)
                        break;
                  end
            end
            Q_old2 = Q_old1;
            Q_old1 = Q;
      end

      #"Done after " << iter << " iterations, Quality is " << Q << std::endl;
      e.mXrot = rotate_givens(e,theta_new);
      cluster_assign(e);

      #output
      e.mQuality = Q;
end
function find_optimal_clustering(data,method)
      Qmax  = -1
      optimal_clustering = nothing
      input = data[:,1:2]
      for i =1:size(data,2)-2
            eigrot=EigenvectorRotation.Config(input, 1)
            Q = EigenvectorRotation.evrot(eigrot)
            if (Q>Qmax)
                  Qmax =Q
                  optimal_clustering =eigrot
            end
            input = [eigrot.mXrot  data[:,i+2]]
      end
      return optimal_clustering
end
function  cluster_assign(e::Config)
      # find max of each row
      max_index_col = zeros(Integer,e.mNumData);
      for i=1:e.mNumData
            col = indmax(abs(e.mXrot[i,:]))
            max_index_col[i] = col;
      end
      #prepare cluster assignments
      for j=1:e.mNumDims # loop over all columns
            for i=1:e.mNumData # loop over all rows
                  if( max_index_col[i] == j )
                        push!(e.mClusters[j],i)
                  end
            end
      end
end
function evqual(e::Config, X)
      #take the square of all entries and find max of each row
      X2 = X.^2
      max_values = maximum(X2,2)
      #compute cost
      for i=1:e.mNumData
            X2[i,:] = X2[i,:] ./ max_values[i];
      end
      J = 1.0 - (sum(X2)/e.mNumData -1.0)/e.mNumDims;
      return J
end
function evqualitygrad(e::Config, theta, angle_index::Integer)
      #build V,U,A
      V = gradU(e,theta, angle_index);

      U1 = build_Uab(e,theta, 1,angle_index);
      U2 = build_Uab(e,theta, angle_index+1,e.mNumAngles);

      A = e.mX*U1*V*U2;

      V=nothing
      U1=nothing;
      U2=nothing

      # rotate vecs according to current angles
      Y = rotate_givens(e,theta);

      #find max of each row
      max_values = zeros(e.mNumData);
      max_index_col = zeros(Integer,e.mNumData)
      for i=1:e.mNumData
            col = indmax(abs(Y[i,:]))
            max_values[i] = Y[i,col];
            max_index_col[i] = col;
      end

      #compute gradient
      dJ=0
      for j=1:e.mNumDims # loop over all columns
            for i=1:e.mNumData # loop over all rows
                  tmp1 = A[i,j] * Y[i,j] / (max_values[i]*max_values[i]);
                  tmp2 = A[i,max_index_col[i]] * (Y[i,j]*Y[i,j]) / (max_values[i]*max_values[i]*max_values[i]);
                  dJ += tmp1-tmp2;
            end
      end
      return 2*dJ/e.mNumData/e.mNumDims;

end

function rotate_givens(e::Config, theta)
      G = build_Uab(e,theta, 1, e.mNumAngles);
      Y = e.mX*G;
      return Y
end

function build_Uab(e::Config, theta,  a::Integer, b::Integer)
      #set Uab to be an identity matrix
      Uab = eye(e.mNumDims,e.mNumDims);
      if( b < a )
            return Uab;
      end
      for  k=a:b
            tt = theta[k];
            for  i=1:e.mNumDims
                  u_ik = Uab[i,e.ik[k]] * cos(tt) - Uab[i,e.jk[k]] * sin(tt);
                  Uab[i,e.jk[k]] = Uab[i,e.ik[k]] * sin(tt) + Uab[i,e.jk[k]] * cos(tt);
                  Uab[i,e.ik[k]] = u_ik;
            end
      end
      return Uab
end
function gradU(e::Config, theta,k::Integer)
      V = zeros(e.mNumDims,e.mNumDims)
      V[e.ik[k],e.ik[k]] = -sin(theta[k]);
      V[e.ik[k],e.jk[k]] = cos(theta[k]);
      V[e.jk[k],e.ik[k]] = -cos(theta[k]);
      V[e.jk[k],e.jk[k]] = -sin(theta[k]);
      return V
end

#Spectral Rotation versus K-Means in Spectral Clustering
function spectral_rotation_svd(Q)
        tic()
      G = (rand(size(Q)).>0.5)*1.0
      iterations = 0
      cambio = true
      while cambio && iterations < 15
            println(iterations)
            cambio=false
            (U,S,V) = svd(Q*G')
            R = U*V'
            for i=1:size(G,2)
                  min_val  = 99999.0
                  min_j = 0
                  for j=1:size(R,2)
                        val = vecnorm(Q[:,i] - R[:,j])^2
                        if (val < min_val)
                              min_val =val
                              min_j = j
                        end
                  end
                  if  G[min_j,i] == 0
                    cambio = true
                    G[:,i] = 0
                    G[min_j,i] = 1
                  end

            end
            iterations=iterations+1
      end

      return G
      #return sum(G.*float(collect(1:size(G,2))'),2)
end
=#
end
