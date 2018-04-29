module Extras
export map_to_01, embedded_image, tweets_output, get_line_segments, digit_features, NMI
using InfoZIP
using Colors
using StringEncodings
using TextAnalysis
using Clustering
url_regexp = r"http[^\s-]*"
function health_tweets(;number_of_tweets=10000)
    local file_path = joinpath(Pkg.dir("SpectralClustering"),"docs","data", "Health-News-Tweets.zip")
    local zipFile = InfoZIP.open_zip(file_path);
    local documents = []
    for (filename, data) in zipFile
        if (startswith(filename, "Health-Tweets/"))
            try 
                data = decode(data, "ISO8859-15")
            catch
            end
            for ln in split(data,'\n')
                local b = split(ln,'|')
                if length(b)>3
                    b[3]=string(b[3:end]...)
                    deleteat!(b,4:length(b))
                end
                if (length(b)==3)
                    b[3] = replace(String(b[3]),url_regexp,"")
                    push!(documents, StringDocument(b[3]))
                end
            end
        end
    end

    local tweets_indices = sample(1:length(documents),min(number_of_tweets,length(documents)),replace=false)
    local corpus = Corpus(documents[tweets_indices])
    prepare!(corpus,  strip_html_tags |  strip_corrupt_utf8  | strip_case | strip_articles | strip_prepositions | strip_pronouns |   strip_punctuation | strip_numbers | strip_non_letters | strip_stopwords     ,skip_words=Set{AbstractString}(["rt","com","nhs","via","cmp","day","amp","net","msn","health","ebola","rss","study","health", "cancer","care","risk","help","patients"]))
    remove_frequent_terms!(corpus,0.95)
    update_lexicon!(corpus)
    

    return corpus
end
function digit_features()
    local path1= joinpath(Pkg.dir("SpectralClustering"),"docs","data", "mfeat-fac")
    local path2= joinpath(Pkg.dir("SpectralClustering"),"docs","data", "mfeat-fou")
    return (readdlm(path1)', readdlm(path2)')

end
function walker()
    
    local file_path = joinpath(Pkg.dir("SpectralClustering"),"docs","data", "tracks.Aug24.zip")
    local a		= InfoZIP.open_zip(file_path)["tracks.24Aug.txt"];
    local n		= length(a)
    local st		= 1
    local at_end	= false
    local datos	= Dict{Integer,Any}()
    local pos		= 1
    local id		= 1
    while !at_end
        pos	= search(a,"TRACK.R",pos)
        if length(pos)>0
            pos		= pos[end]+1
            next	= search(a,"TRACK.R",pos)
            next	= next[end]+1
            datos[id] = []
            while (pos < next)
                m	   = match(r"\[(\d+) (\d+) (\d+)\]", a,pos)
                time   = parse(Int,m.captures[3])
                push!(datos[id], [ parse(Int,m.captures[1]), parse(Int,m.captures[2]), time])
                pos				= m.offset +1
            end
            id					= id +1
        else
            at_end				= true
        end
    end
    return datos
end
function embedded_image(img_size, embedded_patterns)
    local ep        = mapslices(map_to_01,embedded_patterns,1)
    img_clusterized = zeros(RGB, img_size)
    i = 1
    for I in CartesianRange(img_size)
        img_clusterized[I] = RGB(ep[i,:]...)
        i+=1
    end
    return img_clusterized
end
function map_to_01(x::Vector)
	m = 1/(maximum(x)-minimum(x))
	b = -m*minimum(x)
	return m.*x .+b
end
function tweets_output(algo_output, algo_names, docu_term_m)
    output      = Dict{String,Any}("name"=>"result", "children"=>[]);
    for j=1:length(algo_output) 
        local d_algo_j   = Dict{String,Any}("name"=>algo_names[j],"children"=>[])
        push!(output["children"],d_algo_j)
        for i in unique(assignments(algo_output[j]))
            local d_cluster_i    = Dict{String,Any}("name"=>string("Cluster ",i), "children"=>[])
            push!(d_algo_j["children"], d_cluster_i)
            tweet_indices = find(assignments(algo_output[j]).==i)
            mean_tweet    = vec(mean(docu_term_m.dtm[tweet_indices,:],1)) 
            mean_tweet    = mean_tweet ./ sum(mean_tweet) 
            valid_indices = sortperm(mean_tweet, rev=true)[1:15] 
            d_cluster_i["children"] = map(x->Dict("name"=>docu_term_m.terms[x], "size"=>mean_tweet[x]), valid_indices)
        end  
    end
    return output
end
"""
https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf
"""
function NMI(c1::Vector{T1}, gt::Vector{T2}) where T1 where T2
    local N = length(gt)

    local labels_gt = unique(gt)
    local labels_cluster = unique(c1)
    
    local P_gt = [sum(gt.==j)/N for j in labels_gt] 
    local E_gt = sum(-P_gt.*log.(P_gt+eps()))

    local P_cluster = [sum(c1.==j)/N for j in labels_cluster]
    local E_cluster =sum(-P_cluster.*log.(P_cluster+eps()))

    local conditional_entropy = 0

    for i=1:length(labels_cluster)
        local c = labels_cluster[i]
        local indices_of_c = find(c1.==c)
        local N_c = length(indices_of_c)
        local P_given_c = [ sum(gt[indices_of_c].==l)/N_c for l in labels_gt]
        local E_given_c = -P_cluster[i]*sum(P_given_c.*log.(P_given_c+eps()))
        conditional_entropy = conditional_entropy + E_given_c
    end
    local MI =  E_gt - conditional_entropy
    return (2*MI)/(E_gt + E_cluster)
    
end    






module Trajectories
using InfoZIP
using Plots
using Extras
export Trajectory,
    distance,
    start_frame,
    add_point!,
    distance_between_extremes,
    trajectory_confidence,
    simplify,
    TrajectoryDist,
    representative,
    plot_trajectories,
    LineSegment,
    OwnedLineSegment,
    bounding_box,
    calculate_sse,
    AbstractLineSegment,
    angle_distance,
    center,
    angle,
    vector,
    parallel_distance,
    get_line_segments
type Trajectory
    start_frame::Int64
    end_frame::Int64
    points::Vector{Vector{Float32}}
    confidence::Float32
end
function get_line_segments()
    data = Extras.walker()
    line_segments = LineSegment[]
    for k in keys(data)
        for j=1:length(data[k])-1
            p1 = [data[k][j][1]; data[k][j][2]]
            p2 = [data[k][j+1][1]; data[k][j+1][2]]
            if norm(p1-p2) > 5*eps()
                t  = data[k][j][3]
                push!(line_segments, LineSegment(p1,p2,t))
            end
        end
    end
    return line_segments
end

function Trajectory()
    return Trajectory(0,0,[],0.0)
end
abstract type AbstractLineSegment end
struct LineSegment <: AbstractLineSegment
    start_point::Vector{Float32}
    end_point::Vector{Float32}
    time::Integer  
end
import Base.==
function ==(l1::LineSegment, l2::LineSegment)
    return (norm(l1.start_point - l2.start_point) < 2*eps() ) && (norm(l1.start_point - l2.start_point) < 2*eps())
end
struct OwnedLineSegment <: AbstractLineSegment
    start_point::Vector{Float32}
    end_point::Vector{Float32}
    time::Integer  
    trajectory_id::Integer
end
function OwnedLineSegment(start_point::Vector{Float32},end_point::Vector{Float32}, time::Integer)
    return OwnedLineSegment(start_point,end_point,time, -1)
end
import Base.getindex
function getindex(s::AbstractLineSegment, i::Integer)
    if i==1
        return s.start_point
    elseif i==2
        return s.end_point
    else
        error("There is no index 3")
    end
end

import Base.length
function length(l::AbstractLineSegment)
    return norm(l.end_point - l.start_point)
end
function Trajectory(start_frame::Integer)
    return Trajectory(start_frame,start_frame,[],-1)
end
function Trajectory(start_frame::Integer,end_frame::Integer)
    return Trajectory(start_frame,end_frame,[],-1)
end
function add_point!(t::Trajectory, frame::Integer,x::AbstractFloat, y::AbstractFloat)
    if x>0 && y>0
        if length(t.points) == 0
            t.start_frame = frame
        end
        ne_point  = vec([x,y])
        if length(t.points) == 0 || (norm(t.points[end] - ne_point)> 0.1)
            push!(t.points,ne_point)
        end
    end
end
function distance_between_extremes(L)
    return norm(L[1]-L[2])
end
function distance_between_extremes(t::Trajectory)
    return norm(t.points[1]-t.points[end])
end


function project_point_to_line_segment(A,B,p)
    # get dot product of e1, e2
    e1 = vec([B[1]- A[1], B[2] - A[2]]);
    e2 =  vec([p[1] - A[1], p[2] - A[2]]);
    val = dot(e1, e2);
    # get squared length of e1
    len2 = e1[1] * e1[1] + e1[2] * e1[2];
    p = vec([(A[1] + (val * e1[1]) / len2),
             (A[2] + (val * e1[2]) / len2)]);
    return p
end

function projection(Li, Lj)
    
    si = Li[1]
    ei = Li[2]
    sj = Lj[1]      
    ej = Lj[2] 
    u1 = dot(sj-si,ei-si)/(norm(ei-si)^2)
    u2 = dot(ej-si,ei-si)/(norm(ei-si)^2)      
    return (si + u1*(ei-si), si + u2*(ei-si))   
    
end

function angle_distance(si,ei,sj,ej)
    alpha = acosd(max(min(dot(ei - si,ej-sj)/(norm(ei-si)*norm(ej-sj)),1.0),-1.0))
    lj =norm(sj-ej)
    if alpha < 90
        return lj * sind(alpha)
    else
        return lj
    end
end
function angle_distance(L1::Vector,L2::Vector)
    return angle_distance(L1[1],L1[2],L2[1],L2[2])
end
function angle_distance(L1,L2)
    return angle_distance(L1.start_point, L1.end_point, L2.start_point, L2.end_point)
end
function perpendicular_distance(ps,pe,LJ1,LJ2)
    
    if ps!=nothing
        dp1 = norm(ps-LJ1)
    else
        dp1 = 99999
    end
    if pe!=nothing
        dp2 = norm(pe-LJ2)
    else
        dp2= 99999
    end

    
    return  (dp1^2 + dp2^2)/(dp1+dp2+0.000001)
    
end
function perpendicular_distance(ps,pe,Lj)
    return perpendicular_distance(ps,pe,Lj[1],Lj[2])
end
function perpendicular_distance(Li,Lj)
    return perpendicular_distance(Li[1],Li[2],Lj[1],Lj[2])
end

function parallel_distance(ps,pe,si,ei)
    if (ps!=nothing)
        dps1 = norm(ps-si)
        dps2 = norm(ps-ei)
    else
        dps1 = 99999
        dps2 = 99999
    end
    if pe!=nothing
        dpe1 = norm(pe-si)
        dpe2 = norm(pe-ei)
    else
        dpe1 = 99999
        dpe2 = 99999
    end
    val = min(min(dps1,dps2),min(dpe1,dpe2))
    return val
end

function parallel_distance(ps,pe,Li)
    return parallel_distance(ps,pe,Li[1],Li[2])
end
function distance(L1,L2,w=vec([1 1 1]),verbose=false)
    if (L1==L2)
        return 0
    end
    lengthL1 = distance_between_extremes(L1)
    lengthL2 = distance_between_extremes(L2)
    if (lengthL1 > lengthL2)
        Li=L1
        Lj=L2
    else
        Li=L2
        Lj=L1
    end  
    (ps, pe) = projection(Li, Lj)    
    
    pad =  parallel_distance(ps,pe,Li)
    ped = perpendicular_distance(ps,pe,Lj)  
    ang = angle_distance(Li,Lj)
    return w[1]*pad+w[2]*ped+w[3]*ang
end

function mdl_par(t::Trajectory, ps,pe)
    LH   = log2(norm(t.points[ps]-t.points[pe]))
    return LH
end
function mdl_nopar(t::Trajectory, ps,pe)
    LH = 0
    for k=ps:pe-1
        LH   = LH + log2(norm(t.points[k]-t.points[k+1]))
    end

    LDH = 0
    for j=ps:pe-1
        for k=ps:pe-1
            LDH1 = perpendicular_distance(t.points[j],t.points[j+1],t.points[k],t.points[k+1])
            LDH2 =   angle_distance(t.points[j],t.points[j+1],t.points[k],t.points[k+1])
            LDH = LDH + log2(LDH1+0.00000001)+ log2(LDH2+0.00000001)
        end
    end
    return LH + LDH
end
function vector{T<:AbstractLineSegment}(s1::T)
    return s1.end_point - s1.start_point 
    
end
import Base.angle
function angle{T<:AbstractLineSegment}(s1::T)
    v = s1.end_point - s1.start_point 
    ang =  atand(v[2]/v[1])
    if v[1] < 0 
        return 180 + ang
    elseif v[2] < 0
        return 360 + ang
    end
    return ang
end
function center{T<:AbstractLineSegment}(s1::T)
    return (s1.end_point + s1.start_point )/2.0
end
function bounding_box{T<:AbstractLineSegment}(line_seg::T)
    ymin = min(line_seg.start_point[2], line_seg.end_point[2]) 
    ymax = max(line_seg.start_point[2], line_seg.end_point[2]) 
    xmin = min(line_seg.start_point[1], line_seg.end_point[1]) 
    xmax = max(line_seg.start_point[1], line_seg.end_point[1]) 
    return (convert(Vector{Float64},vec([xmin ymin])), convert(Vector{Float64},vec([xmax ymax])))

end
function bounding_box{T<:AbstractLineSegment}(line_seg::T, s::AbstractFloat)
    bb = bounding_box(line_seg)
    center = vec((bb[1] + bb[2]) / 2.0)
    width  = (bb[2][1] - bb[1][1]) * s
    height = (bb[2][2] - bb[1][2]) * s
    
    return (center- vec([width height]), center+vec([width height]))
end
function bounding_box{T<:AbstractLineSegment}(line_seg::T, eps::Vector)
    bb = bounding_box(line_seg)
    bb[1][1] = bb[1][1] - eps[1]
    bb[1][2] = bb[1][2] - eps[2]
    
    bb[2][1] = bb[2][1] + eps[1]
    bb[2][2] = bb[2][2] + eps[2]
    
    return bb
end
function simplify(t::Trajectory)
    tn = Trajectory(t.start_frame,t.end_frame)
    push!(tn.points,t.points[1])
    startindex=1
    len=1
    while startindex + len <= length(t.points)
        currindex = startindex + len
        costpar = mdl_par(t,startindex,currindex)
        costnopar = mdl_nopar(t,startindex,currindex)

        if costpar  > costnopar
            push!(tn.points,t.points[currindex])
            startindex = currindex 
            len=1
        else
            len=len+1
        end
    end

    push!(tn.points,t.points[end])
    return tn
end

function float_equals(v1::AbstractFloat,v2::AbstractFloat)

    return abs(v1-v2) < 0.00000001
end
function average_direction{T<:AbstractLineSegment}(list::Vector{T})
    av_dir = [0,0]
    for v in list
        dir = (v[2]-v[1])
        
        av_dir = av_dir + dir
    end
    return av_dir/length(list)
end
function average_coordinate{T<:AbstractLineSegment}(x,list::Vector{T})
    y = 0
    for v in list
        y = y + v[1][2]
    end
    y=y/length(list)
    return [x,y]
end
function align_x_to_segments{T<:AbstractLineSegment}(list::Vector{T})
    avg_dir = average_direction(list)
    angle_respect_x   = acos(dot(avg_dir, convert(Vector{Float32},[1.0, 0.0]))/norm(avg_dir))
    rot_matrix        = [cos(angle_respect_x) sin(angle_respect_x);
                         -sin(angle_respect_x) cos(angle_respect_x)]
    rotated_points = []
    tree = SpatialIndex.RTree(2)  
    

    j=1
    for s in list
        p1 = rot_matrix * s[1]
        p2 = rot_matrix * s[2]
        p = LineSegment(p1,p2)
        bb = Trajectories.bounding_box(p)
        push!(rotated_points,p1)
        push!(rotated_points,p2) 
        SpatialIndex.add_point!(tree,j, bb[1],bb[2]);
        j=j+1
    end
    return (rot_matrix,rotated_points, tree)
end


function is_between(a,c,b)
    minx = min(a,b)
    maxx = max(a,b)
    return c >= minx && c<= maxx
    
end
function representative{T<:AbstractLineSegment}(list::Vector{T}, MinLns, gamma )
    rep                 = Vector{Vector{Float32}}()
    (r_matrix,new_list,tree) = align_x_to_segments(list)
    r_matrix_inv        = inv(r_matrix)
    new_list_sort       = sortperm(new_list, lt=(p1,p2)->(p1[1]<p2[1]))
    previous_point = vec([-Inf -Inf])
    for i=1:length(new_list_sort)
        p  = new_list[new_list_sort[i]]
        p1 = map(Float64,vec([p[1]; -15000]       ))
        p2 =  map(Float64,vec([p[1]; 15000]       ))
        trajectories_sharing_x = intersects(tree,p1,p2)
        if (length(trajectories_sharing_x) >= MinLns)
            diff = p[1] - previous_point[1]
            if diff >= gamma                
                ff = mean([new_list[r][2] for  r in  trajectories_sharing_x])
                av_coordinate = [p[1],   ff]
                new_point = r_matrix_inv*av_coordinate                
                push!(rep,new_point)                               
                previous_point = av_coordinate
            end
        end
    end
    return Trajectory(1,50,rep,-1)
end

import Plots.plot
Plots.@recipe function f(segments::Vector{Trajectories.LineSegment}, colors)
    x= Float32[]
    y= Float32[]
    for s in segments       
        push!(x,s.start_point[1])
        push!(x,s.end_point[1])
        push!(y,s.start_point[2])
        push!(y,s.end_point[2])
        push!(x, NaN)
        push!(y, NaN)
    end
    linecolor := colors
    (x,y)
end
end
end
