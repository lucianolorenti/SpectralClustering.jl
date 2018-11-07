export get_element,
       number_of_patterns,
       number_of_pixels,
       get_pixel,
       get_element!,
       spatial_position,
      pattern_dimension,
    get_element,
    assign!

using ColorTypes

function pattern_dimension(X)
  return length(get_element(X,1))
end





"""
```julia
function get_element!(o::Matrix,  img::Matrix{C}, i::Vector{Integer}) where C<:Colorant

```
"""
function get_element!(o::D, img::Matrix{C}, i::Vector{<:Integer}) where D<:AbstractArray  where C<:Colorant
    rows,cols   = ind2sub(size(img),i)
    values = getindex.((img,), rows,cols)
    @inbounds o[1,:] = cols
    @inbounds o[2,:] = rows
    N = length(C)
    component = N >= 3 ? (comp1, comp2, comp3, alpha) : (comp1, alpha)
    for j=1:length(C)
        @inbounds o[2+j,:] = component[j].(values)
    end
end

"""
```julia
function assign!(vec::T, val::C) where T<:AbstractArray where C<:Colorant
```
This function assigns the components of the color component val to a vector v
"""
function assign!(vec::T, val::C) where T<:AbstractArray where C<:Colorant
   N = length(C)
   component = N >= 3 ? (comp1, comp2, comp3, alpha) : (comp1, alpha)
    for j=1:length(C)
        @inbounds vec[j] = component[j](val)
    end
end

"""
```@julia
get_element!{T<:AbstractArray}(vec::T,  img::Matrix{Gray}, i::Integer)
```

Return through```vec``` the intensity image element  [x,y, i], where \$x,y\$ are the spatial
position of the pixel and the value i of the pixel \$(x,y)\$.
"""
function get_element!(vec::T,  img::Matrix{C}, i::Integer) where T<:AbstractArray where C<:Colorant
    ind    = spatial_position(img,i)
    @inbounds vec[1] = ind[2]
    @inbounds vec[2] = ind[1]
    assign!(view(vec,3:length(vec)), img[i])
end



"""
```
function get_element( img::Matrix{RGB}, i::Vector) 
```
"""
function get_element(img::Matrix{T}, i::Vector)  where T<:Colorant
    m = zeros(length(T)+2,length(i))
    get_element!(m,img,i)
    return m
end

function get_element(img::Matrix{T}, i::Integer)  where T<:Colorant
   m = zeros(length(T)+2)
    get_element!(m,img,i)
    return m
end

"""
```@julia
number_of_patterns{T<:Any}(X::Matrix{T,3})
```

Return the number of pixels in the image
"""
number_of_patterns(X::Matrix{T}) where T<:Colorant = size(X,1)*size(X,2)

"""
```@julia
 spatial_position(X::Matrix, i::Int)
```
Returns the sub indexes from the linear index ```i```
"""
function spatial_position(img::Matrix, i::Int)
  return   ind2sub(size(img),i)
end

function spatial_position(img::Matrix, i::Vector{M}) where M<:Integer
  return   ind2sub(size(img),i)
end


get_element(X::T,i) where T<:AbstractArray= view(X,:,i)
number_of_patterns(X::T) where T<:AbstractArray    = size(X,2)

get_element(X::Vector,i) = X[i]
number_of_patterns(X::Vector)     = length(X)

number_of_patterns(X::Array{T, 3}) where T = size(X, 2) * size(X, 3)
"""
```
function get_element(data::Array{T, 3}, i::Vector)  where T<:Number
```
"""
function get_element(data::Array{T, 3}, i::Vector)  where T<:Number
    m = zeros(size(data,1)+2,length(i))
    get_element!(m, data, i)
    return m
end

get_element(data::Array{T, 3}, i::Integer)  where T<:Number = get_element(data, [i])
get_element!(o::AbstractArray, data::Array{T,3}, i::Integer) where T<:Number = get_element!(o, data, [i])

function get_element!(o::AbstractArray, data::Array{T,3}, i::Vector) where T<:Number
    (_, nr, nc) = size(data)
    cart_indices = CartesianIndices((nr,nc))[i]
    @inbounds for (i, pos) in enumerate(cart_indices)
        o[3:end, i] = data[:, pos[1], pos[2]]
        o[1,i] = pos[2]
        o[2,i] = pos[1]
    end
end

