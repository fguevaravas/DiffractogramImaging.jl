using DiffractogramImaging, PCC, Test, LinearAlgebra, FFTW
⊗ = LinearAlgebra.kron
DI = DiffractogramImaging
relerr(a,b) = norm(a-b)/norm(a)

function test_Q(p,q)
 Q = DI.linmap_Q(p...,q...)
 x = randn(prod(p)*prod(q)) + 1im*randn(prod(p)*prod(q))
 y = Q*x
 x_ = Q'*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

function test_F(n)
 F = DI.linmap_F(n...)
 x = randn(prod(n)) + 1im*randn(prod(n))
 y = F*x
 x_ = F'*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

function test_pcc2d(p,q)
 n = p.*q
 k = randn(n...,p...)
 k̂ = DI.pcc2d_to_BlockDiagonal(k)
 Q = DI.linmap_Q(p...,q...)
 K = Q'*k̂*Q

 x = randn(n...) + im*randn(n...)
 y = K*vec(x)

 k̂_ = PCC.pcc2d_gemv_blkm_prep(k)
 y_ = vec(pcc2d_gemv_blkm(x,k̂_))
 @test relerr(y,y_) ≈ 0 atol=1e-10
end

function test_M_solve(n1,n2,N,α,β,γ)
 M = DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ)
 Minv = DI.linmap_M_blktrdiag_solve(n1,n2,N,α,β,γ)

 x = randn(N*n1*n2) + 1im*randn(N*n1*n2)

 y = M*x
 x_ = Minv*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
 y = Minv*x
 x_ = M*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

# we build M explicitly so only do this for small sizes
function test_M(n1,n2,N,α,β,γ)
 D1 = fft(DI.rk_periodic_derivative(1,(n1,n2)))
 D2 = fft(DI.rk_periodic_derivative(2,(n1,n2)))
 L = diagm(vec(abs2.(D1) + abs2.(D2)))
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 M = α^2 * I(N) ⊗ L + β^2 * LN ⊗ I(n1*n2) + γ^2*I
 M_ =  DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ)
 x = randn(N*n1*n2) + 1im*randn(N*n1*n2)
 y = M*x
 y_ = M_*x
 @test relerr(y,y_) ≈ 0 atol=1e-10
end


test_cases = [
  (n=(10,11),p=(1,1)),
  (n=(11,9),p=(1,1)),
  (n=(27,32),p=(3,4)),
  (n=(32,27),p=(4,3)),
  (n=(32,32),p=(32,32)),
  (n=(32,27),p=(32,27)) ]

# Actually run the tests
@testset "Q, F, pcc2d" begin
 for c in test_cases
  q = c.n .÷ c.p
  println("sizes $c")
  test_Q(c.p,q)
  test_F(c.n)
  test_pcc2d(c.p,q)
 end
end

@testset "testing M" begin
 test_M(12,12,4,1,2,3)
 test_M(12,12,4,3,2,1)
 test_M(24,12,4,3,2,1)
 test_M(12,24,4,3,2,1)
 test_M(12,24,1,3,2,1)
end

@testset "testing M solve" begin
 test_M_solve(12,24,1,3,2,1)
 test_M_solve(12,12,4,1,2,3)
 test_M_solve(12,12,4,3,2,1)
 test_M_solve(24,12,4,3,2,1)
 test_M_solve(12,24,4,3,2,1)
end
