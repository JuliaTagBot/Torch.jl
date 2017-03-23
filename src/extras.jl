THArray(args...) = THArray{Float64}(args...)

Base.size(xs::THArray) = ntuple(i -> size(xs, i), ndims(xs))

THArray(data::AbstractArray) = copy!(THArray{eltype(data)}(size(data)), data)
