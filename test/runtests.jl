using DiffractogramImaging, PCC, Test, LinearAlgebra, FFTW, BlockDiagonals
⊗ = LinearAlgebra.kron
DI = DiffractogramImaging
relerr(a,b) = norm(a-b)/norm(a)

function test_Q(p,q)
 Q = DI.linmap_Q(p...,q...)
 x = randn(prod(p)*prod(q)) + 1im*randn(prod(p)*prod(q))
 y = Q*x
 x_ = Q'*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
 y = Q'*x
 x_ = Q*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

function test_F(n)
 F = DI.linmap_F(n...)
 x = randn(prod(n)) + 1im*randn(prod(n))
 y = F*x
 x_ = F'*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
 y = F'*x
 x_ = F*y
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

# use only for small sizes: tests convolution theorem
function test_conv(n)
 k = randn(n...)+1im*randn(n...)
 x = randn(n...)+1im*randn(n...)
 F = DI.linmap_F(n...)
 xk1 = F'*((F*vec(x)).*vec(fft(k))) # even if F is unitary the diag matrix is the same

 # naive circular convolution
 xk2 = zeros(ComplexF64,n...)
 for i1=1:n[1],i2=1:n[2]
  for j1=1:n[1],j2=1:n[2]
   xk2[i1,i2] += k[mod1(i1-j1+1,n[1]),mod1(i2-j2+1,n[2])]*x[j1,j2]
  end
 end

 @test relerr(xk1,vec(xk2)) ≈ 0 atol=1e-10

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

# only tests on small sizes
function test_rk_to_pcc2d(p,q)
 n = p .* q
 k = randn(n...) + im*randn(n...)
 k̃ = DI.rk_to_pcc2d(k,p...,q...)
 k̂ = DI.pcc2d_to_BlockDiagonal(k̃)
 F = DI.linmap_F(n...)
 Q = DI.linmap_Q(p...,q...)
 Fk̂ = Diagonal(vec(fft(k))) 
 K1 = F'*Fk̂*F
 K2 = Q'*k̂*Q
 x = randn(prod(n)) + 1im*randn(prod(n))
 y1 = K1*x
 y2 = K2*x
 @test relerr(y1,y2) ≈ 0 atol=1e-10
end

# we build E explicity, so do this only for small sizes
function test_E(p1,p2,q1,q2,N,α=1,β=2,γ=3,ρ=5)
 n1 = p1*q1; n2=p2*q2
 # generate fake a
 a = [ BlockDiagonal([ randn(p1*p2,p1*p2) + im*randn(p1*p2,p1*p2) for j=1:q1*q2]) for i=1:N]
 E = DI.BlockDiagonal_E(a,α,β,γ,ρ,p1,p2,q1,q2)

 # dense M
 D1 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(1,(n1,n2))),p1,p2,q1,q2)
 D2 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(2,(n1,n2))),p1,p2,q1,q2)
 D̂1 = DI.pcc2d_to_BlockDiagonal(D1)
 D̂2 = DI.pcc2d_to_BlockDiagonal(D2)
 L = D̂1'*D̂1 + D̂2'*D̂2
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 M = ρ*(α^2*I(N)⊗L + β^2*LN⊗I(n1*n2) + γ^2*I)

 A = hcat(a...) # calculates a dense A
 x = randn(n1*n2) + im*randn(n1*n2)
 y  = E*x
 y_ = x + A*(M\(A'*x))

 @test relerr(y,y_) ≈ 0 atol=1e-10
end

# check both representations for M. Full matrices so only small sizes!!
function test_M_full(p1,p2,q1,q2,N,α=1,β=2,γ=3)
 n1 = p1*q1; n2 = p2*q2
 # dense transforms!
 Q = Matrix(DI.linmap_Q(p1,p2,q1,q2))
 F = Matrix(DI.linmap_F(n1,n2))

 # M with block diagonals
 D1 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(1,(n1,n2))),p1,p2,q1,q2)
 D2 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(2,(n1,n2))),p1,p2,q1,q2)
 D̂1 = DI.pcc2d_to_BlockDiagonal(D1)
 D̂2 = DI.pcc2d_to_BlockDiagonal(D2)
 L = D̂1'*D̂1 + D̂2'*D̂2
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 M1 = (I(N) ⊗ Q') * (α^2*I(N)⊗L + β^2*LN⊗I(n1*n2) + γ^2*I) * (I(N) ⊗ Q)

 # M with FFTs
 D1 = fft(DI.rk_periodic_derivative(1,(n1,n2)))
 D2 = fft(DI.rk_periodic_derivative(2,(n1,n2)))
 L = diagm(vec(abs2.(D1) + abs2.(D2)))
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 M2 = (I(N) ⊗ F') * (α^2 * I(N) ⊗ L + β^2 * LN ⊗ I(n1*n2) + γ^2*I)  * (I(N) ⊗ F)

 x = randn(N*n1*n2) + im*randn(N*n1*n2)
 
 y1 = M1*x
 y2 = M2*x

 @test relerr(y1,y2) ≈ 0 atol=1e-10
end

# another test for E: checks equality including all the
# unitary transformations. ONLY small sizes
function test_E_full(p1,p2,q1,q2,N,α=1,β=2,γ=3,ρ=1)
 n1 = p1*q1; n2=p2*q2
 # generate fake a
 a = [ BlockDiagonal([ randn(p1*p2,p1*p2) + im*randn(p1*p2,p1*p2) for j=1:q1*q2]) for i=1:N]
 Ẽ = DI.BlockDiagonal_E(a,α,β,γ,ρ,p1,p2,q1,q2)

 # dense transforms!
 Q = Matrix(DI.linmap_Q(p1,p2,q1,q2))
 F = Matrix(DI.linmap_F(n1,n2))
 

 E = Q'*Ẽ*Q

 D1 = fft(DI.rk_periodic_derivative(1,(n1,n2)))
 D2 = fft(DI.rk_periodic_derivative(2,(n1,n2)))
 L = diagm(vec(abs2.(D1) + abs2.(D2)))
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 M̂ = ρ*(α^2 * I(N) ⊗ L + β^2 * LN ⊗ I(n1*n2) + γ^2*I)

 M = (I(N) ⊗ F') * M̂ * (I(N) ⊗ F)

 A = Q' * hcat(a...) * (I(N) ⊗ Q)  
 
 E_ = I + A*(M\A')

 x =randn(n1*n2) + 1im*rand(n1*n2)
 y = E*x
 y_ = E_*x

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
#if false #################################
@testset "testing Q" begin
 for c in test_cases
  q = c.n .÷ c.p
  test_Q(c.p,q)
 end
end

@testset "testing F" begin
 for c in test_cases
  q = c.n .÷ c.p
  test_F(c.n)
 end
end

@testset "testing pcc2d" begin
 for c in test_cases
  q = c.n .÷ c.p
  test_pcc2d(c.p,q)
 end
end

@testset "testing M" begin
 test_M(12,24,1,3,2,1)
 test_M(12,24,2,3,2,1)
 test_M(12,24,4,3,2,1)
 test_M(24,12,4,3,2,1)
end

@testset "testing M solve" begin
 test_M_solve(12,24,1,3,2,1)
 test_M_solve(12,24,2,3,2,1)
 test_M_solve(12,24,4,3,2,1)
 test_M_solve(24,12,4,3,2,1)
end

@testset "testing conv thm" begin
 test_conv((10,12))
 test_conv((12,12))
 test_conv((12,9))
end

@testset "testing rk_to_pcc2d" begin
 for c in test_cases
  q = c.n .÷ c.p
  test_rk_to_pcc2d(c.p,q)
 end
end

@testset "testing E" begin
  test_E(3,4,2,3,1)
  test_E(3,4,2,3,2)
  test_E(3,4,2,3,4)
end

@testset "test E full" begin
  test_E_full(3,4,2,3,1)
  test_E_full(3,4,2,3,2)
  test_E_full(3,4,2,3,4)
end
#end ##########################
