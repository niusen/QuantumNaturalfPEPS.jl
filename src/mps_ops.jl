#using ITensors: sim!, deprecate_make_inds_match!, check_hascommoninds
# From abstractmps.jl


using ITensors



using ITensors: sim!

function _log_or_not_dot(
    M1::MPST, M2::MPST, loginner::Bool; dag=true, make_inds_match::Bool=true
)::Number where {MPST<:ITensors.AbstractMPS}
    N = length(M1)
    if length(M2) != N
        throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
    end

    M1dag = dag ? ITensors.dag(M1) : M1

    sim!(linkinds, M1dag)

    siteindsM1dag = ITensors.siteinds(all, M1dag)
    siteindsM2 = ITensors.siteinds(all, M2)

    same_num_siteinds = length(M1) == length(M2) &&
        all(n -> length(ITensors.siteinds(M1, n)) == length(ITensors.siteinds(M2, n)), 1:length(M1))

    if any(n -> length(n) > 1, siteindsM1dag) ||
       any(n -> length(n) > 1, siteindsM2) ||
       !same_num_siteinds
        make_inds_match = false
    end

    if make_inds_match
        ITensors.replace_siteinds!(M1dag, siteindsM2)
    end

    O = M1dag[1] * M2[1]

    if loginner
        normO = norm(O)
        log_inner_tot = log(normO)
        O ./= normO
    end

    for j in eachindex(M1)[2:end]
        O = (O * M1dag[j]) * M2[j]

        if loginner
            normO = norm(O)
            log_inner_tot += log(normO)
            O ./= normO
        end
    end

    if loginner
        if !isreal(O[]) || real(O[]) < 0
            log_inner_tot += log(complex(O[]))
        end
        return log_inner_tot
    end

    dot_M1_M2 = O[]

    if !isfinite(dot_M1_M2)
        @warn "The inner product (or norm²) you are computing is very large " *
              "($dot_M1_M2). You should consider using `lognorm` or `loginner` instead, " *
              "which will help avoid floating point errors. For example if you are trying " *
              "to normalize your MPS/MPO `A`, the normalized MPS/MPO `B` would be given by " *
              "`B = A ./ z` where `z = exp(lognorm(A) / length(A))`."
    end

    return dot_M1_M2
end

