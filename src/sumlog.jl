"""
    sumlog(x::AbstractArray{T}; dims=:)

Computes `sum(log.(x); dims)`.

When `float(T) <: AbstractFloat`, this is done by a faster algorithm than
`sum(log, x; dims)`, calling `log` just once (or once per output element, with `dims`).
This works by representing the `j`th element of `x` as ``x_j = a_j  2^{b_j}``,
allowing us to write
```math
\\sum_j \\log{x_j} = \\log(\\prod_j a_j) + \\log{2} \\sum_j b_j

Relative accuracy is very similar to that of `sum(log, x)`,
except when the answer is very close to zero.
For example `x = fill(prevfloat(1.0), 1000)` has `sum(log, x) ≈ -1.11e-13` but `sumlog(x) == 0.0`.
```
"""
sumlog(x::AbstractArray{T}; dims=:) where T = _sumlog(float(T), dims, x)

function _sumlog(::Type{T}, ::Colon, x) where {T<:AbstractFloat}
    if T<:Base.Math.IEEEFloat && length(x) < 129
        # Around size 30 this path is 5x faster. By size 1000, it usually fails to Inf. 
        # y = prod(x) do xj
        #     xj < 0 && Base.Math.throw_complex_domainerror(:log, xj)
        #     Float64(xj)
        # end
        # Checking errors on a separate pass is slightly slower at size 10, but faster at 30, 100
        # any(<(0), x) && Base.Math.throw_complex_domainerror(:log, -1.0)
        # ... and this way is faster than `any`:
        foreach(x) do xj
            xj < 0 && Base.Math.throw_complex_domainerror(:log, xj)
        end
        # Not using Float32 important, could use promote_type(T, Float64) to allow BigFloat
        y = prod(Float64, x; init=1.0)
        isfinite(y) && return T(log(y))  # Not sure whether log ought to be before isfinite, e.g. for zero, TODO
    end
    # Note that `init=(1.0, 0)` is the neutral element, so this `mapreduce` is safe for GPUs.
    sig, ex = mapreduce(_sumlog_op, x; init=(one(T), 0)) do xj
        float_xj = float(xj)
        _significand(float_xj), _exponent(float_xj) 
    end
    return log(sig) + IrrationalConstants.logtwo * T(ex)
end

# It's possible that this case should call `sumlog!(y, x)` as that might be useful by itself.
function _sumlog(::Type{T}, dims, x) where {T<:AbstractFloat}
    sig_ex = mapreduce(_sumlog_op, x; dims=dims, init=(one(T), 0)) do xj
        float_xj = float(xj)
        _significand(float_xj), _exponent(float_xj) 
    end
    map(sig_ex) do (sig, ex)
        log(sig) + IrrationalConstants.logtwo * T(ex)
    end
    # Suggestion here is that broadcasting lets this handle scalar case too:
    # https://github.com/JuliaStats/LogExpFunctions.jl/pull/48#discussion_r867387426
    # return log.(first.(sig_ex)) .+ IrrationalConstants.logtwo .* T.(last.(sig_ex))
    # But it doesn't work, as first.(::Tuple) doesn't pick the first element.
end

# Fallback: `float(T)` is not always `<: AbstractFloat`, e.g. complex, dual numbers or symbolics
_sumlog(::Type, dims, x) = sum(log, x; dims)

@inline function _sumlog_op((sig1, ex1), (sig2, ex2))
    sig = sig1 * sig2
    # Instead of explicit errors, you might be able to ensure `sig < 0` if any `x_j < 0`, that might be faster?
    # sig = ifelse(sig2<0, sig2, sig1 * sig2)  # slow
    # sig *= ifelse(signbit(sig2), NaN, 1.0)   # slow
    ex = ex1 + ex2
    # Significands are in the range [1,2), so multiplication will eventually overflow
    if sig > floatmax(typeof(sig)) / 2
        ex += _exponent(sig)
        sig = significand(sig)
    end
    return sig, ex
end

# It's important to check sign, else sumlog([1,-2,-3]) can work, and tidier to do it in this function.
# Perhaps there is a faster way, returning `NaN` instead might be OK.
_significand(x::AbstractFloat) = x<0 ? Base.Math.throw_complex_domainerror(:log, x) : significand(x)

# The exported `exponent(x)` checks for `NaN` and `-0.0` etc, this function doesn't. Fine as `sig` keeps track.
_exponent(x::Base.IEEEFloat) = Base.Math._exponent_finite_nonzero(x)
_exponent(x::AbstractFloat) = Int(exponent(x))  # e.g. for BigFloat, this gives errors. Copy its ccall in here?

"""
    sumlog(x)
    sumlog(f, x, ys...)

For any iterator which produces `AbstractFloat` elements,
this can use `sumlog`'s fast reduction strategy.

Signature with `f` is equivalent to `sum(log, map(f, x, ys...))`,
but is performed without intermediate allocations.

Does not accept a `dims` keyword.
"""
sumlog(f, x) = sumlog(Iterators.map(f, x))
sumlog(f, x, ys...) = sumlog(f(xy...) for xy in zip(x, ys...))

# Iterator version, uses the same `_sumlog_op`, should be the same speed.
function sumlog(x)
    iter = iterate(x)
    if isnothing(iter)
        # Empty case. Possibly you could write float(sum(x)), maybe init=false? TODO
        T = Base._return_type(first, Tuple{typeof(x)})
        return T <: Number ? zero(float(T)) : 0.0
    end
    x1 = float(iter[1])
    x1 isa AbstractFloat || return sum(log, x)  # Should ideally not restart iterator here, TODO
    sig, ex = _significand(x1), _exponent(x1)
    nonfloat = zero(x1)
    iter = iterate(x, iter[2])
    while iter !== nothing
        xj = float(iter[1])
        if xj isa AbstractFloat
            sig, ex = _sumlog_op((sig, ex), (_significand(xj), _exponent(xj)))
        else
            # Allow for iterator to change type mid-stream, without restarting.
            nonfloat += log(xj)
        end
        iter = iterate(x, iter[2])
    end
    return log(sig) + IrrationalConstants.logtwo * oftype(sig, ex) + nonfloat
end
