module IncrementalSpectralClustering
export EigenUpdater, add_edge, add_vertex
using Graph

gr = Graph

type EigenUpdater
  oracle::Function
  connect_with
  k
  val
  vec
  W
  D
end
#graph         = gr.build(data,KNNGraphParameters(k,metric,oracle))
function EigenUpdater(graph,
                      connect_with::Function,
                      oracle::Function,
                      k,
                      number_of_vectors = 4)

  (W,D )  =calculate_W_and_d(graph)
  println(find(sum(W,1).<0.0001))
  L = D-W
  
  (evals,evecs) = eigs(Symmetric(L),nev=number_of_vectors,which=:SM,maxiter=5000)
  #(m_eval,m_evec) = eigs(L,nev=2,which=:LM,maxiter=5000)

  #m_evec       =  real(Dinv * m_evec)
  evals         = real(evals)
  #m_eval         = real(m_eval)
  idx = sortperm(evals)
  evals = evals[idx]
  evecs = evecs[:,idx]

  #evals= vcat(evals[idx], m_eval)
  #evecs = hcat(evecs[:,idx],m_evec)
  return EigenUpdater(oracle,connect_with,k,evals,evecs,W,D)
end
type Deltas
  lambda
  qi
end
import Base.+
function +(d1::Deltas,d2::Deltas)
  return Deltas(d1.lambda+d2.lambda,d1.qi+d2.qi)
end
 function add_edge(conf::EigenUpdater,i,j,w)
   n_vertexes = size(conf.W,1)
   u = sparsevec(Dict(i => 1, j => -1), n_vertexes)
   v = spdiagm((full(sparsevec(Dict(i => 1, j => 1), n_vertexes)),),0)

   delta_L = u*u'*w
   delta_D = w*v



   W=conf.W
   D=conf.D

   n  = sort(collect(
                     union(
                       Set{Integer}(find(W[i,:].>0)),
                       Set{Integer}(find(W[j,:].>0)),
                       Set{Integer}([i,j])
                       )))
   deltas = Array{Deltas}(length(conf.val))

   for k=1:length(conf.val)
     delta_lambda_prev = Inf
     delta_lambda = 0
     qi_prev = ones(size(W,1))
     qi = zeros(size(W,1))
     while (abs(delta_lambda_prev -delta_lambda) > 0.00000000001) && (norm(qi_prev[n]-qi[n]) > 0.00000000001)

         delta_lambda_prev = delta_lambda
         qi_prev = qi
         a = ((conf.vec[i,k]-conf.vec[j,k]^2)^2 - conf.val[k]*(conf.vec[i,k]^2+conf.vec[j,k]^2))
         b = (conf.vec[i,k] - conf.vec[i,k]) * (qi[i] - qi[j]) - conf.val[k]*(conf.vec[i,k]*qi[i] + conf.vec[j,k]*qi[j])
         c = w*(qi[i]^2 + qi[j]^2)
         d= 0
         for idx in n
             d = d +  (conf.vec[idx,k] * D[idx,idx] *qi[idx] )
         end
         delta_lambda = w*((a+b) /    ( c+d +1))
         K = (D-W) - conf.val[k]*D
         K = K[:,n]
         h = real((delta_lambda * D + conf.val[k]*delta_D - delta_L)*conf.vec[:,k])
         #(val, vec) = eig(full(K'*K))
         #val = real(val)
         #vec = real(vec)
         #val[val.>0] = 1./val[val.>0]
        #f =  (vec'*diagm(val)*vec)
        #println(typeof(f))
        #println(typeof(K))
        #println(typeof(h))
         qi[n] = (pinv(full(K'*K)))*K' * h
   end
   if k==2
     #println(conf.val[k])
     #println(delta_lambda)
     #println(conf.val[k])
   end

   conf.val[k] = real(conf.val[k]) + real(delta_lambda)
   conf.vec[n,k] = conf.vec[n,k] + qi[n]
   deltas[k] = Deltas([conf.val[k]+abs(delta_lambda) abs(delta_lambda) ], qi)

   end
   W[i,j]=w
   W[j,i]=w
   D[i,i]+=w
   D[j,j]+=w
   return deltas

   #Graph.add_edge!(conf.graph,i,j,w)
end
  function add_vertex(conf::EigenUpdater, new_data )
   #idx_new_vertex = Graph.add_vertex!(conf.graph)
   n_vertexes = size(conf.W,1)
   idx_new_vertex = n_vertexes + 1
   W_1 = spzeros(n_vertexes+1,n_vertexes+1)
   W_1[1:n_vertexes,1:n_vertexes] = conf.W
   conf.W=W_1
   D_1 = spzeros(n_vertexes+1,n_vertexes+1)
   D_1[1:n_vertexes,1:n_vertexes] = conf.D
   conf.D=D_1

   conf.vec= [conf.vec; zeros(size(conf.vec,2))']
   indexes  = conf.connect_with(new_data)
   deltas=nothing
   for i in indexes
     if (i!=idx_new_vertex)
        ww = conf.oracle(new_data,i);
        if  ( ww > 0.0001)
          current_delta = add_edge(conf, Int(i), Int(idx_new_vertex), ww)
          if (deltas == nothing)
            deltas=current_delta
          else
            for i=1:length(deltas)
               deltas[i] = deltas[i] + current_delta[i]
            end
          end

        end
     end
   end

   return deltas
 end
end
