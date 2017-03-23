THArray(args...) = THArray{Float64}(args...)

Base.size(xs::THArray) = ntuple(i -> size(xs, i), ndims(xs))

THArray{T}(data::AbstractArray) where T = copy!(THArray{T}(size(data)), data)

THArray(data::AbstractArray) = THArray{eltype(data)}(data)

Base.similar(xs::THArray) = typeof(xs)(size(xs))

Base.:+{T}(xs::THArray{T}, ys::THArray{T}) = add!(xs, ys, out = similar(xs))
Base.:-{T}(xs::THArray{T}, ys::THArray{T}) = sub!(xs, ys, out = similar(xs))
