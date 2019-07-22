using CuArrays
using GPUArrays

Flux.gpu(a::SubArray) = CuArray{Float32}(a)


function Base.repeat(src::CuArray{T, 2}, x::Int) where T
    m, n = size(src)
    dest = CuArray{T, 2}(undef, m*x, n)

    function kernel(state, dest, src, m)
        idx = @cartesianidx(dest, state)
        dest[idx...] = src[(idx[1]-1) % m + 1, idx[2]]
        return
    end
    gpu_call(kernel, dest, (dest, src, size(src, 1)))
    dest
end