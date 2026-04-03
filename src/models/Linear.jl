# linear OU dX_t = A X_t dt + √2 B dW_t
# A has to have eigenvalues with only negative real part for stationary measures to exist

module LinearSDE

using LinearAlgebra, MatrixEquations, StaticArrays

export linear_drift!, linear_noise!, analytic_potential

@inline function linear_drift(u::SVector{S, T}, A::SMatrix{S, S, T}, t) where {S, T}
    # A is 2x2, u is 2-element SVector
    du1 = A[1,1]*u[1] + A[1,2]*u[2]
    du2 = A[2,1]*u[1] + A[2,2]*u[2]
    
    du[1] = du1
    du[2] = du2
end

@inline function linear_noise(u::SVector{S, T}, A::SMatrix{S, S, T}, t) where {S, T}
    # B is 2x2, assuming additive diagonal noise for this OU process
    # Adjust indexing if B is intended to be a vector or non-diagonal
    s2 = sqrt(2.0f0)
    du[1] = s2 * B[1,1]
    du[2] = s2 * B[2,2]
end

function analytic_potential(A, B)
    S = lyapc(TEST_A, 2*TEST_B*TEST_B')
    V(x) = 1/2 * (x ⋅ (S\x))
    return V
end

end;