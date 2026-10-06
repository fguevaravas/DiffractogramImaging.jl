using DiffractogramImaging, PCC, Test, LinearAlgebra, FFTW
using BlockDiagonals, LinearMaps
⊗ = LinearAlgebra.kron
DI = DiffractogramImaging
relerr(a,b) = norm(a-b)/norm(a)

# Calculates Neumann 1D Laplacian of size N.  The case N=1 is not well defined
# but this function returns [1]
function neumann_laplacian(N)
 LN = diagm(0=>2*ones(N),-1=>-ones(N-1),1=>-ones(N-1))
 LN[1,1]=1; LN[N,N]=1; # use Neumann conditions
 return LN
end

function test_Q(p,q)
 Q = DI.linmap_Q(p...,q...)
 x = randn(prod(p)*prod(q)) + 1im*randn(prod(p)*prod(q))
 x_ = Q'*Q*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
 x_ = Q*Q'*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

function test_F(n)
 F = DI.linmap_F(n...)
 x = randn(prod(n)) + 1im*randn(prod(n))
 x_ = F'*F*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
 x_ = F*F'*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

# only small sizes
function test_Q_N(p,q,N)
 Q = DI.linmap_Q_N(p...,q...,N)
 Q_ = I(N) ⊗ DI.linmap_Q(p...,q...)
 @test norm(Matrix(Q) - Matrix(Q_)) ≈ 0 atol=1e-10
end

# only small sizes
function test_F_N(n,N)
 F = DI.linmap_F_N(n...,N)
 F_ = I(N) ⊗ DI.linmap_F(n...)
 @test norm(Matrix(F) - Matrix(F_)) ≈ 0 atol=1e-10
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
 x_ = Minv*M*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
 x_ = M*Minv*x
 @test relerr(x,x_) ≈ 0 atol=1e-10
end

# we build M explicitly so only do this for small sizes
function test_M(n1,n2,N,α,β,γ)
 D1 = fft(DI.rk_periodic_derivative(1,(n1,n2)))
 D2 = fft(DI.rk_periodic_derivative(2,(n1,n2)))
 L = diagm(vec(abs2.(D1) + abs2.(D2)))
 LN = neumann_laplacian(N)
 M = α^2 * I(N) ⊗ L + (N>1)*β^2 * LN ⊗ I(n1*n2) + γ^2*I
 M_ =  DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ)
 x = randn(N*n1*n2) + 1im*randn(N*n1*n2)
 y = M*x
 y_ = M_*x
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
 LN = neumann_laplacian(N)
 M̂1 = α^2*I(N)⊗L + (N>1)*β^2*LN⊗I(n1*n2) + γ^2*I
 M1 = (I(N) ⊗ Q') * M̂1 * (I(N) ⊗ Q)

 # M with FFTs
 M2 = Matrix((I(N) ⊗ F') * DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ) * (I(N) ⊗ F))

 # check matvec is same
 x = randn(N*n1*n2) + im*randn(N*n1*n2)
 y1 = M1*x
 y2 = M2*x
 @test relerr(y1,y2) ≈ 0 atol=1e-10

 # check positive definiteness
 M1inv = cholesky(Hermitian(M1),check=false)
 M2inv = cholesky(Hermitian(M2),check=false)
 @test issuccess(M1inv)
 @test issuccess(M2inv)
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
 E = Matrix(DI.BlockDiagonal_E(a,α,β,γ,ρ,p1,p2,q1,q2))

 # dense M as block diag matrix
 D1 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(1,(n1,n2))),p1,p2,q1,q2)
 D2 = DI.rk_to_pcc2d((DI.rk_periodic_derivative(2,(n1,n2))),p1,p2,q1,q2)
 D̂1 = DI.pcc2d_to_BlockDiagonal(D1)
 D̂2 = DI.pcc2d_to_BlockDiagonal(D2)
 L = D̂1'*D̂1 + D̂2'*D̂2
 LN = neumann_laplacian(N)
 M = ρ*(α^2*I(N)⊗L + (N>1)*β^2*LN⊗I(n1*n2) + γ^2*I)

 A = hcat(a...) 
 x = randn(n1*n2) + im*randn(n1*n2)
 y  = E*x
 y_ = x + A*(M\(A'*x))

 @test relerr(y,y_) ≈ 0 atol=1e-10
end

# another test for E: checks equality including all the
# unitary transformations. ONLY small sizes
function test_E_full(p1,p2,q1,q2,N,α=1,β=2,γ=3,ρ=5)
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
 LN = neumann_laplacian(N)
 M̂ = ρ*(α^2 * I(N) ⊗ L + (N>1)*β^2 * LN ⊗ I(n1*n2) + γ^2*I)
 M = (I(N) ⊗ F') * M̂ * (I(N) ⊗ F)
 #M = Matrix((I(N) ⊗ F') * DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ) * (I(N) ⊗ F))

 A = Q' * hcat(a...) * (I(N) ⊗ Q)  
 
 E_ = I + A*(M\(A'))

 x =randn(n1*n2) + 1im*rand(n1*n2)
 y = E*x
 y_ = E_*x
 @test relerr(y,y_) ≈ 0 atol=1e-10

 # test positive definiteness
 Einv_ = cholesky(Hermitian(E_),check=false)
 Einv  = cholesky(Hermitian(E) ,check=false)
 @test issuccess(Einv_)
 @test issuccess(Einv)

end

# this is only for small sizes
function test_R(n1,n2,N,α=1,β=2,γ=3)
 R1 = DI.linmap_R(n1,n2,N,α,β,γ)
 # periodic difference matrix
 function Dp(n)
  res = zeros(n,n)
  for i=1:n-1
   res[i,[i,i+1]] = [-1 1]
  end
  res[n,[n,1]] = [-1,1]
  return res
 end

 # difference matrix
 function D(n)
  res =  zeros(n-1,n)
  for i=1:n-1
   res[i,[i,i+1]] = [-1 1]
  end
  return res
 end
 
 D1 = I(N) ⊗ I(n2) ⊗ Dp(n1)
 D2 = I(N) ⊗ Dp(n2) ⊗ I(n1)
 DN = D(N) ⊗ I(n1*n2)

 R2 = [α*D1 
       α*D2
       β*DN
       γ*I ]

 # check R
 x = randn(n1*n2*N) + 1im*randn(n1*n2*N)
 y1 = R1*x
 y2 = R2*x
 @test relerr(y1,y2) ≈ 0 atol=1e-10

 # check R'
 x = randn(n1*n2*(4N-1)) + 1im*randn(n1*n2*(4N-1))
 y1 = R1'*x
 y2 = R2'*x
 @test relerr(y1,y2) ≈ 0 atol=1e-10

 # check R'*R = M
 F = Matrix(DI.linmap_F(n1,n2)) # dense!!
 M̂ = DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ) 
 M = Matrix((I(N) ⊗ F') * M̂  * (I(N) ⊗ F))

 @test norm(R2'*R2 - M)≈0 atol=1e-10
end

# this is only for small sizes!!
function test_augmented_ls_solver(p1,p2,q1,q2,N,α=1,β=2,γ=3,ρ=5)
 @assert ρ>0
 n1 = p1*q1; n2=p2*q2
 # generate fake a
 a = [ BlockDiagonal([ randn(p1*p2,p1*p2) + im*randn(p1*p2,p1*p2) for j=1:q1*q2]) for i=1:N]
 # dense transforms!
 Q = Matrix(DI.linmap_Q(p1,p2,q1,q2))
 F = Matrix(DI.linmap_F(n1,n2))
 A = Matrix(Q' * hcat(a...) * (I(N) ⊗ Q))
 R = Matrix(DI.linmap_R(n1,n2,N,α,β,γ))
 M = Matrix((I(N) ⊗ F') * DI.linmap_M_blktrdiag(n1,n2,N,α,β,γ) * (I(N) ⊗ F))
 E_ = Matrix(DI.BlockDiagonal_E(a,α,β,γ,ρ,p1,p2,q1,q2))
 B = [ A
       sqrt(ρ)*R  ]

 E = I+A*inv(ρ*M)*A'
 @test norm(E-Q'*E_*Q)≈0 atol=1e-10

 # test SMW for pseudoinverse
 BBi1 = inv(B'*B) # expensive
 
 BBi2 = inv(ρ*M) - inv(ρ*M)*A'*inv(E)*A*inv(ρ*M)
 @test norm(BBi1- BBi2)≈0 atol=1e-10

 pinvB1 = BBi1*B'
 pinvB2 = Matrix(DI.linmap_augmented_ls_solver(p1,p2,q1,q2,a,α,β,γ,ρ))
 @test norm(pinvB1 - pinvB2)≈0 atol=1e-10
end

########################################################################

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


@testset "testing M full" begin
  test_M_full(3,4,2,3,1)
  test_M_full(3,4,2,3,2)
  test_M_full(3,4,2,3,3)
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


@testset "testing R" begin
 test_R(10,11,2)
 test_R(11,10,4)
end

@testset "multifreq versions of Q and F" begin
 cases = [
  (p=(2,2),q=(3,3),N=3),
  (p=(2,2),q=(3,3),N=3),
  (p=(3,2),q=(5,3),N=1) ]
 for c ∈ cases
  n = c.p .* c.q
  test_F_N(n,c.N)
  test_Q_N(c...)
 end
end
#end ##################

@testset "augment LS solver" begin
  α,β,γ,ρ = 1,2,3,5
  test_augmented_ls_solver(3,4,2,3,2,α,β,γ,ρ)
  test_augmented_ls_solver(3,4,2,3,4,α,β,γ,ρ)
end

