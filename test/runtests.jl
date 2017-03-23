using Torch
using Base.Test

xs = rand(5,3)

th = THArray(xs)

@test xs == th
