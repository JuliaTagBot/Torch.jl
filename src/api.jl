struct THArray{T,N} #<: AbstractArray{T,N}
  ptr::Ptr{Void}
end

for N = 1:4
  @eval function (::Type{THArray{Float64,$N}})(size::NTuple{$N,Integer})
    ptr = ccall(:THDoubleTensor_newWithSize4d, Ptr{Void},
                (Clong, Clong, Clong, Clong), $([i ≤ N ? :(size[$i]) : -1 for i = 1:4]...))
    THArray{Float64,$N}(ptr)
  end
end

THArray{Float64}(size::NTuple{N,Integer}) where N = THArray{Float64,N}(size)

function Base.size(xs::THArray{Float64}, dim::Integer)
  @assert 1 ≤ dim ≤ ndims(xs)
  ccall(:THDoubleTensor_size, Clong, (Ptr{Void}, Cint), xs.ptr, dim-1)
end

function Base.size(xs::THArray{Float64}, dim::Integer)
  @assert 1 ≤ dim ≤ ndims(xs)
  ccall(:THDoubleTensor_size, Clong, (Ptr{Void}, Cint), xs.ptr, dim-1)
end

function Base.collect(xs::THArray{Float64})
  data = ccall(:THDoubleTensor_data, Ptr{Float64}, (Ptr{Void},), xs.ptr)
  copy(unsafe_wrap(Array, data, size(xs)))
end

function Base.fill!(xs::THArray{Float64}, x::Real)
  ccall(:THDoubleTensor_fill, Void, (Ptr{Void}, Float64), xs.ptr, x)
  return xs
end
