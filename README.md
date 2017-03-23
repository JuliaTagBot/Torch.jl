# Torch.jl

Torch.jl is a wrapper around the array data structure provided by [Torch](https://github.com/torch/torch7). Torch's implementations are easy to use with standard Julia array interfaces, don't try to do anything magical, run on GPUs, and are blazing fast, making this an attractive way to speed up array code in Julia without code changes.

This package does not aim to provide higher-level features of Torch like support for neural networks, only the basic array type. Other packages can provide this and simply operate with Torch arrays (ideally for free).

```julia
xs = THArray(rand(5, 2))
ys = THArray(rand(5, 2))

xs + ys # Add to THArrays

Torch.add!(xs, ys) # Add to `xs` in place
```

Current limitations:

* No attempt at a build step. You need to build the Torch dll (go to [this folder](https://github.com/torch/torch7/tree/master/lib/TH), `cmake`, `make`) and change the hard-coded path in this package.
* Not all of the Torch array API is wrapped. There's enough of the array interface for it to work generally, but not everything will be optimally fast.
* Torch's approach to array storage, as well as things like broadcasting, is pretty different to Julia's. It will take some thinking to get the two to mesh well.
