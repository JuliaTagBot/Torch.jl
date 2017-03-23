module Torch

include("api.jl")
include("extras.jl")

function __init__()
  Libdl.dlopen(joinpath(@__DIR__, "..", "deps", "libTH.dylib"))
end

end # module
