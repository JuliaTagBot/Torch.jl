mutable struct THArray{T,N} <: AbstractArray{T,N}
  ptr::Ptr{Void}
end

THVector{T} = THArray{T,1}
THMatrix{T} = THArray{T,2}

litsym(xs...) = Expr(:quote, Symbol(xs...))

for (T, th) in [(Float64, :Double),
                (Float32, :Float),
                (Float16, :Half),
                (Int64,   :Long),
                (Int32,   :Int),
                (Int16,   :Short),
                (UInt8,   :Byte),
                (Int8,    :Char)]
  THTensor_(s...) = litsym(:TH, th, :Tensor_, s...)
  for N = 1:4
    @eval begin
      function (::Type{THArray{$T,$N}})(size::NTuple{$N,Integer})
        ptr = ccall($(THTensor_(:newWithSize, N, :d)), Ptr{Void},
                    ($(ntuple(_->Clong, N)...),), $(ntuple(i->:(size[$i]),N)...))
        xs = THArray{$T,$N}(ptr)
        finalizer(xs, free)
        return xs
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

    function free(xs::THArray{$T})
      ccall($(THTensor_(:free)), Void, (Ptr{Void},), xs.ptr)
    end

    function Base.size(xs::THArray{$T}, dim::Integer)
      @assert 1 ≤ dim ≤ ndims(xs)
      ccall($(THTensor_(:size)), Clong, (Ptr{Void}, Cint), xs.ptr, dim-1)
    end

    function Base.fill!(xs::THArray{$T}, x::Real)
      ccall($(THTensor_(:fill)), Void, (Ptr{Void}, $T), xs.ptr, x)
      return xs
    end

    function copy!(xs::THArray{$T}, ys::THArray{$T})
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:copy)), Void, (Ptr{Void}, Ptr{Void}),
            xs.ptr, ys.ptr)
      return xs
    end

    function add!(xs::THArray{$T}, ys::THArray{$T}; out = xs)
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:cadd)), Void, (Ptr{Void}, Ptr{Void}, $T, Ptr{Void}),
            out.ptr, xs.ptr, 1, ys.ptr)
      return out
    end

    function sub!(xs::THArray{$T}, ys::THArray{$T}; out = xs)
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:csub)), Void, (Ptr{Void}, Ptr{Void}, $T, Ptr{Void}),
            out.ptr, xs.ptr, 1, ys.ptr)
      return out
    end

    function mul!(xs::THArray{$T}, ys::THArray{$T}; out = xs)
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:cmul)), Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}),
            out.ptr, xs.ptr, ys.ptr)
      return out
    end

    function div!(xs::THArray{$T}, ys::THArray{$T}; out = xs)
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:div)), Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}),
            out.ptr, xs.ptr, ys.ptr)
      return out
    end

    function pow!(xs::THArray{$T}, ys::THArray{$T}; out = xs)
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:pow)), Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}),
            out.ptr, xs.ptr, ys.ptr)
      return out
    end

    function LinAlg.dot(xs::THVector{$T}, ys::THVector{$T})
      @assert size(xs) == size(ys)
      ccall($(THTensor_(:dot)), $T, (Ptr{Void}, Ptr{Void}),
            xs.ptr, ys.ptr)
    end

  end
end
