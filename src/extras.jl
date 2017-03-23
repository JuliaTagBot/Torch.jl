THArray(args...) = THArray{Float64}(args...)

# TODO: remove when we support AbstractArray
Base.eltype{T}(::THArray{T}) = T
Base.ndims{T,N}(::THArray{T,N}) = N

Base.size(xs::THArray) = ntuple(i -> size(xs, i), ndims(xs))
