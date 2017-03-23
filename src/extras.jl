THArray(args...) = THArray{Float64}(args...)

Base.size(xs::THArray) = ntuple(i -> size(xs, i), ndims(xs))

THArray{T}(data::AbstractArray) where T = copy!(THArray{T}(size(data)), data)

THArray(data::AbstractArray) = THArray{eltype(data)}(data)

Base.similar(xs::THArray) = typeof(xs)(size(xs))

import Base: +, -

(xs::THArray{T} + ys::THArray{T}) where T = add!(xs, ys, out = similar(xs))
(xs::THArray{T} - ys::THArray{T}) where T = sub!(xs, ys, out = similar(xs))
