using DiffractogramImaging, PCC, Test, LinearAlgebra, FFTW
DI = DiffractogramImaging
relerr(a,b) = norm(a-b)/norm(a)

function test_Q(p,q)
 Q = DI.linmap_Q(p...,q...)
 x = randn(prod(p)*prod(q))
 y = Q*x
 x_ = Q'*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

function test_F(n)
 F = DI.linmap_F(n...)
 x = randn(prod(p)*prod(q))
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
 y = K*x

 k̂_ = pcc2d_gemv_blkm_prep(k)
 y_ = pcc2d_gemv_blkm(x,k̂_)
 @test relerr(y,y_) ≈ 0 atol=1e-10
end

test_cases = [
  (n=(27,32),p=(1,1)),
  (n=(32,27),p=(1,1)),
  (n=(27,32),p=(3,4)),
  (n=(32,27),p=(4,3)),
  (n=(32,32),p=(32,32)),
  (n=(32,27),p=(32,27)) ]

for c in test_cases
 q = c.n .÷ c.p
 @testset "sizes $c" begin
  test_Q(c.p,q)
  test_F(c.n)
  test_pcc2d(c.p,q)
 end
end
