THArray{T}{T,N}(size::NTuple{N,Integer}) = THArray{T,N}(size)
THArray{T}{T,N}(size::Vararg{Integer,N}) = THArray{T}(size)

THArray(args...) = THArray{Float64}(args...)

THArray{T}{T}(data::AbstractArray) = copy!(THArray{T}(size(data)), data)

THArray(data::AbstractArray) = THArray{eltype(data)}(data)

Base.size(xs::THArray) = ntuple(i -> size(xs, i), ndims(xs))

Base.similar(xs::THArray) = typeof(xs)(size(xs))

# TODO: promotion for THArrays of different types

import Base: +, -

(xs::THArray{T} + ys::THArray{T}) where T = add!(xs, ys, out = similar(xs))
(xs::THArray{T} - ys::THArray{T}) where T = sub!(xs, ys, out = similar(xs))
