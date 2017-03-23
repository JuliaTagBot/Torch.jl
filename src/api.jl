struct THArray{T,N} <: AbstractArray{T,N}
  ptr::Ptr{Void}
end

litsym(xs...) = Expr(:quote, Symbol(xs...))

for (T, th) in [(Float64, :Double), (Float32, :Float)]
  THTensor_(s...) = litsym(:TH, th, :Tensor_, s...)
  for N = 1:4
    @eval begin
      function (::Type{THArray{$T,$N}})(size::NTuple{$N,Integer})
        ptr = ccall($(THTensor_(:newWithSize, N, :d)), Ptr{Void},
                    ($(ntuple(_->Clong, N)...),), $(ntuple(i->:(size[$i]),N)...))
        THArray{$T,$N}(ptr)
      end
      function Base.getindex(xs::THArray{$T,$N}, idx::Vararg{Integer,$N})
        @assert all(1 .≤ idx .≤ size(xs))
        ccall($(THTensor_(:get, N, :d)), $T,
              (Ptr{Void}, $(ntuple(_->Clong, N)...)),
              xs.ptr, $(ntuple(i->:(idx[$i]-1),N)...))
      end
      function Base.setindex!(xs::THArray{$T,$N}, x::Real, idx::Vararg{Integer,$N})
        @assert all(1 .≤ idx .≤ size(xs))
        ccall($(THTensor_(:set, N, :d)), Void,
              (Ptr{Void}, $(ntuple(_->Clong, N)...), $T),
              xs.ptr, $(ntuple(i->:(idx[$i]-1),N)...), x)
        return x
      end
    end
  end
  @eval begin

    THArray{$T}(size::NTuple{N,Integer}) where N = THArray{$T,N}(size)

    function Base.size(xs::THArray{$T}, dim::Integer)
      @assert 1 ≤ dim ≤ ndims(xs)
      ccall($(THTensor_(:size)), Clong, (Ptr{Void}, Cint), xs.ptr, dim-1)
    end

    function Base.size(xs::THArray{$T}, dim::Integer)
      @assert 1 ≤ dim ≤ ndims(xs)
      ccall($(THTensor_(:size)), Clong, (Ptr{Void}, Cint), xs.ptr, dim-1)
    end

    function Base.fill!(xs::THArray{$T}, x::Real)
      ccall($(THTensor_(:fill)), Void, (Ptr{Void}, $T), xs.ptr, x)
      return xs
    end

  end
end
