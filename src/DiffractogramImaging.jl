module DiffractogramImaging
 using LinearMaps, FFTW, LinearAlgebra, BlockDiagonals,Random
 ⊗ = LinearAlgebra.kron 
 fftwComplex = FFTW.fftwComplex
 
 """
  Encapsulate the unitary transformation that makes
  a PCC block diagonal as a LinearMap
  Inputs:
   T complex type (e.g. ComplexF64)
   p1,p2,q1,q2 dimensions
  Outputs: p1*q1 x p2*q2 LinearMap
 """
 function linmap_Q(::Type{T},p1,p2,q1,q2) where {T<:fftwComplex}
  n1 = p1*q1; n2 = p2*q2;

  function Qmul!(y,x)
   x = reshape(x,p1,q1,p2,q2)
   y = reshape(y,p1,p2,q1,q2)
   x̂ = fft(x,(2,4))/sqrt(q1*q2)
   permutedims!(y,x̂,(1,3,2,4))
   return vec(y)
  end

  function Qmulh!(x,y)
   y = reshape(y,p1,p2,q1,q2)
   x = reshape(x,p1,q1,p2,q2)
   permutedims!(x,y,(1,3,2,4))
   bfft!(x,(2,4))
   x./=sqrt(q1*q2)
   return vec(x)
  end
  return LinearMap{T}(Qmul!,Qmulh!,n1*n2,n1*n2,ismutating=true)
 end
 linmap_Q(p1,p2,q1,q2) = linmap_Q(ComplexF64,p1,p2,q1,q2) 
 
 """
  calculates I(N) ⊗ Q
 """
 function linmap_Q_N(::Type{T},p1,p2,q1,q2,N) where {T<:fftwComplex}
  n1 = p1*q1; n2 = p2*q2;
  function Qmul!(y,x)
   x = reshape(x,p1,q1,p2,q2,N)
   y = reshape(y,p1,p2,q1,q2,N)
   x̂ = fft(x,(2,4))/sqrt(q1*q2)
   permutedims!(y,x̂,(1,3,2,4,5))
   return vec(y)
  end

  function Qmulh!(x,y)
   y = reshape(y,p1,p2,q1,q2,N)
   x = reshape(x,p1,q1,p2,q2,N)
   permutedims!(x,y,(1,3,2,4,5))
   bfft!(x,(2,4))
   x./=sqrt(q1*q2)
   return vec(x)
  end
  return LinearMap{T}(Qmul!,Qmulh!,n1*n2*N,n1*n2*N,ismutating=true)
 end
 linmap_Q_N(p1,p2,q1,q2,N) = linmap_Q_N(ComplexF64,p1,p2,q1,q2,N) 

 """
  Encapsulate the unitary  discrete Fourier transform
  Inputs:
   T complex type (e.g. ComplexF64)
   p1,p2,q1,q2 dimensions
  Outputs: p1*q1 x p2*q2 LinearMap
 """
 function linmap_F(::Type{T},n1,n2) where {T<:fftwComplex}
  function Fmul!(y,x)
   y = reshape(y,n1,n2)
   x = reshape(x,n1,n2)
   y .= x/sqrt(n1*n2)
   fft!(y)
   return vec(y)
  end

  function Fmulh!(y,x)
   y = reshape(y,n1,n2)
   x = reshape(x,n1,n2)
   y .= x/sqrt(n1*n2)
   bfft!(y)
   return vec(y)
  end
  return LinearMap{T}(Fmul!,Fmulh!,n1*n2,n1*n2,ismutating=true)
 end
 linmap_F(n1,n2) = linmap_F(ComplexF64,n1,n2)

 """
 calculates I(N) ⊗ F
 """
 function linmap_F_N(::Type{T},n1,n2,N) where {T<:fftwComplex}
  function Fmul!(y,x)
   y = reshape(y,n1,n2,N)
   x = reshape(x,n1,n2,N)
   y .= x/sqrt(n1*n2)
   fft!(y,(1,2))
   return vec(y)
  end

  function Fmulh!(y,x)
   y = reshape(y,n1,n2,N)
   x = reshape(x,n1,n2,N)
   y .= x/sqrt(n1*n2)
   bfft!(y,(1,2))
   return vec(y)
  end
  return LinearMap{T}(Fmul!,Fmulh!,n1*n2*N,n1*n2*N,ismutating=true)
 end
 linmap_F_N(n1,n2,N) = linmap_F_N(ComplexF64,n1,n2,N)

  """
  block matrix representation of a kernel
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required. Also
   n1 == p1*q1 and n2 == p2*q2

  Outputs:
   k̂ BlockDiagonal representation of operator
 """
 function pcc2d_to_BlockDiagonal(k)
  n1,n2,p1,p2 = size(k)
  @assert (n1%p1==0) && (n2%p2==0)
  q1 = n1÷p1; q2 = n2÷p2;
  k̂ = fft(reshape(k,p1,q1,p2,q2,p1,p2),(2,4))
  k̂ = reshape(permutedims(k̂,(1,3,5,6,2,4)),p1*p2,p1*p2,q1*q2)
  return BlockDiagonal([ k̂[:,:,i] for i=1:q1*q2 ])
 end
 
 """
  a linear map representing the matrix

   R = [ I(N) ⊗ D1
         I(N) ⊗ D2
         DN ⊗ I(n1*n2)
         I(n1*n2*N) ]
  Inputs:
   n1,n2,N dimensions

   R is (4N-1)*n1*n2 x N*n1*n2 
   note that there are no rows associated with DN 
   when N==1
 """
 function linmap_R(n1,n2,N)
  d1 = circshift(1:n1,-1)
  d2 = circshift(1:n2,-1)
  d1t = circshift(1:n1,1)
  d2t = circshift(1:n2,1)
  function mul!(y,x)
   x = reshape(x,n1,n2,N)
   y = reshape(y,n1,n2,4N-1)
   y[:,:,1:N] .= x[d1,:,:] - x
   y[:,:,N.+(1:N)] .= x[:,d2,:] - x
   y[:,:,2N.+(1:N-1)] .= x[:,:,2:N] - x[:,:,1:N-1]
   y[:,:,(3N-1).+(1:N)] .= x
   return vec(y)
  end
  function mulh!(x,y)
   x = reshape(x,n1,n2,N)
   y = reshape(y,n1,n2,4N-1)
   x .= 0
   x .+= y[d1t,:,1:N] - y[:,:,1:N]
   x .+= y[:,d2t,N.+(1:N)] - y[:,:,N.+(1:N)]
   x[:,:,2:N]   .+= y[:,:,2N.+(1:N-1)]
   x[:,:,1:N-1] .-= y[:,:,2N.+(1:N-1)]
   x .+= y[:,:,(3N-1).+(1:N)]
   return vec(x)
  end
  return LinearMap{ComplexF64}(mul!,mulh!,(4N-1)*n1*n2,N*n1*n2,ismutating=true)
 end
 
 """
  identifies the residuals to indices
 """
 residual_indices(n1,n2,N) = [1:N,N.+(1:N),2N.+(1:N-1),(3N-1).+(1:N)]

 """
 filter associated with derivative operator of a n-periodic signal
 along the dimension specified in dim
 """
 function rk_periodic_derivative(::Type{T},dim,n) where {T<:fftwComplex}
    KD = zeros(T,n...)
    N = length(n)
    1 <= dim <= N || throw(ArgumentError("dimension $dim out of range 1:$N"))
    ref   = ntuple(i->1,N)
    refp1 = ntuple(i-> i==dim ? 2 : 1,N)
    KD[ref...]=1
    KD[refp1...]=-1
    return KD
 end
 rk_periodic_derivative(dim,n) = rk_periodic_derivative(ComplexF64,dim,n)


 """
  block tridiagonal symmetric real matrix M = R'*R representation using I ⊗ F
  in this basis, each block is a diagonal matrix of size n1*n2 x n1*n2

  returns a n1 * n2 * N linear map corresponding to:
    I_N ⊗ (D1'*D1 + D2'*D2) + L_N ⊗ I_{n_1 n_2} + I_N ⊗ I_{n_1 n_2} 
 """
 function linmap_M_blktrdiag(n1,n2,N)
  # calculate the weight for periodic Laplacian in spatial variables
  D1 = fft(rk_periodic_derivative(1,(n1,n2)))
  D2 = fft(rk_periodic_derivative(2,(n1,n2)))
  L = vec(abs2.(D1) + abs2.(D2))

  b = (N>1) # to handle case N==1
  # diagonal blocks are all copies of
  d = L .+ (2b + 1)
  # except for the first and last block in wavelength 
  # (due to L_N being Neumann BC Laplacian)
  dn = L .+ (b + 1)
  
  # lower diagonal blocks are not stored, since they are just
  # multiplication by -β^2

  function amul!(y,x)
   y = reshape(y,n1*n2,N)
   x = reshape(x,n1*n2,N)
   if N==1
    y[:,1] .= dn.*x[:,1]
    return vec(y)
   end
   y[:,1] .= dn.*x[:,1] - b*x[:,2]
   for i=2:N-1
    y[:,i] .= -b*x[:,i-1] + d.*x[:,i] - b*x[:,i+1]
   end
   y[:,N] .= -b*x[:,N-1] + dn.*x[:,N]
   return vec(y)
  end
  return LinearMap{ComplexF64}(amul!,amul!,N*n1*n2,N*n1*n2,ismutating=true)
 end

 """
  solves a system with matrix M
 """
 function linmap_M_blktrdiag_solve(n1,n2,N)
  # calculate the weight for periodic Laplacian in spatial variables
  D1 = fft(rk_periodic_derivative(1,(n1,n2)))
  D2 = fft(rk_periodic_derivative(2,(n1,n2)))
  L = vec(abs2.(D1) + abs2.(D2))

  b = (N>1) # to handle case N==1

  # diagonal blocks are all copies of
  d = L .+ (2b + 1)
  # except for the first and last block in wavelength 
  # (due to L_N being Neumann BC Laplacian)
  dn = L .+ (b + 1)

  # lower diagonal blocks are not stored, since they are just
  # multiplication by -b
  
  # compute Cholesky factor M = LL^*
  cd = zeros(n1*n2,N); # diagonal blocks
  cl = zeros(n1*n2,N-1); # lower diagonal  blocks
  cd[:,1] = sqrt.(dn)
  for i=1:N-1
   cl[:,i] = -b./cd[:,i]
   if i<N-1
    cd[:,i+1] = sqrt.(d - cl[:,i].^2)
   else
    cd[:,i+1] = sqrt.(dn - cl[:,i].^2)
   end
  end 

  # to compute action of M^{-1} = L^{-*} L^{-1} 
  function amul!(x,y)
   y = reshape(y,n1*n2,N)
   x = reshape(x,n1*n2,N)

   # forward substitution: solve L u = y for u in place
   x .= y
   x[:,1] ./= cd[:,1]
   for i=2:N
    x[:,i] .-= cl[:,i-1].*x[:,i-1]
    x[:,i] ./= cd[:,i]
   end

   # backward substitution: solve L^* x = u in place
   x[:,N] ./= cd[:,N]
   for i=N-1:-1:1
    x[:,i] .-= cl[:,i].*x[:,i+1]
    x[:,i] ./= cd[:,i]
   end
  end

  return LinearMap{ComplexF64}(amul!,amul!,N*n1*n2,N*n1*n2,ismutating=true)
 end 
 """
  Converts a vector of blocktridiagonal matrices to a linear map.
  All block tridiagonal matrices are assumed to be square and of the same size
  Inputs
   a an array of block tridiagonal matrices
 """
 function linmap_blkdiag_hcat(a)
  N = length(a)
  n = size(a[1],1)
  function amul!(y,x)
   x = reshape(x,n,N)
   y .= 0
   for i=1:N
    y .+= a[i]*x[:,i]
   end
  end
  function amulh!(y,x)
   y = reshape(y,n,N)
   for i=1:N
    y[:,i] .= a[i]'*x
   end
  end
  return LinearMap{ComplexF64}(amul!,amulh!,n,N*n,ismutating=true)
 end
 
 """
  Convert a regular convolution kernel to a periodic kernel
  Inputs
   k  n1 x n2 regular convolution kernel
   k̃  n1 x n2 x p1 x p2 corresponding p1,p2 - periodic kernel
 """
 function rk_to_pcc2d(k,p1,p2,q1,q2)
  n1,n2 = size(k)
  @assert n1==p1*q1 && n2==p2*q2
  k̃ = zeros(eltype(k),n1,n2,p1,p2)
  for i1=1:p1,i2=1:p2
    k̃[:,:,i1,i2] = circshift(k,(i1-1,i2-1))
  end
  return k̃
 end

 const VectorBlockDiag{T} = Vector{BlockDiagonal{T, Matrix{T}}}

 """
  Construct the block matrix representation of the matrix E = I + ρ^-1 A M^{-1} A^*
  (this can be expensive, but only needs to be done once)
 """
 function BlockDiagonal_E(a::Vector{<:BlockDiagonal{T, <:AbstractMatrix{T}}},
                          ρ,p1,p2,q1,q2) where T<:Complex
  n1 = p1*q1; n2 = p2*q2
  N = length(a) 
  @assert all(size(a[1]) .== (n1*n2,n1*n2))
  @assert nblocks(a[1]) == q1*q2

  b = (N>1) # to handle case N==1

  ## Constructing M as PCC by blocks
  # diagonal blocks of M as PCC
  d1 = rk_periodic_derivative(1,(n1,n2))
  d2 = rk_periodic_derivative(2,(n1,n2))

  # put kernel for D1 as a PCC
  D1 = rk_to_pcc2d(d1,p1,p2,q1,q2)
  D2 = rk_to_pcc2d(d2,p1,p2,q1,q2)

  # block matrix rep
  D̂1 = pcc2d_to_BlockDiagonal(D1)
  D̂2 = pcc2d_to_BlockDiagonal(D2)

  # Laplacian in space
  L̂ = D̂1'*D̂1 + D̂2'*D̂2 

  # diagonal blocks are all copies of
  MD = L̂ + (2b + 1)*I
  # except for first and last (due to Neumann BC Laplacian)
  MDn = L̂ + (b + 1)*I

  # offdiagonal blocks are all copies of
  ML = -b*I(n1*n2)

  # Cholesky factorization of M as a PCC
  CD = Vector{Any}(undef,N)     # diagonal blocks
  CL = VectorBlockDiag{T}(undef,N-1)   # lower diagonal blocks
  CD[1]=cholesky(MDn)
  for i=1:N-1
   # note: the following does not work. it seems to be a bug between
   # BlockDiagonals and Diagonal interactions
   #CL[i] = ML/CD[i].U 
   CL[i] = -b*inv(CD[i].U) 
   if i<N-1
    CD[i+1] = cholesky(MD - CL[i]*CL[i]')
   else
    CD[i+1] = cholesky(MDn - CL[i]*CL[i]')
   end
  end 

  # since M = L L^*, we need calculate Y = M^{-1} A^* = L^-* L^-1 A^*
   
  # multiply L^-1 A^* (forward substitution)
  u = VectorBlockDiag{T}(undef,N)
  u[1] = CD[1].L\(a[1]')
  for i=2:N
   u[i] = CD[i].L\(a[i]' - CL[i-1]*u[i-1])
  end

  # multiply by L^* (backward substituion)
  y = VectorBlockDiag{T}(undef,N)
  y[N] = CD[N].U\u[N]
  for i=N-1:-1:1
   y[i] = CD[i].U\(u[i] - CL[i]'*y[i+1])
  end

  # multiply on the left by A
  E = zero(a[1])
  for i=1:N
   E += a[i]*y[i]
  end
  E = I + E/ρ 
  return Hermitian(E)
 end 
 """
  Inputs:
    a block matrix representation of matrix A
    α,β,γ,ρ regularization parameters
    p1,p2,q1,q2 problem size 

  Outputs:
    A linear map equivalent to solving the problem

         min || B x - b ||_2^2
 """
 function linmap_augmented_ls_solver(p1,p2,q1,q2,a,ρ)
  @assert ρ>0
  n1 = p1*q1; n2 = p2*q2
  N = length(a)
  Q = linmap_Q(p1,p2,q1,q2)
  Q_N = linmap_Q_N(p1,p2,q1,q2,N)
  F_N = linmap_F_N(n1,n2,N)
  R = linmap_R(n1,n2,N)
  Minv = F_N'*linmap_M_blktrdiag_solve(n1,n2,N)*F_N
  E = BlockDiagonal_E(a,ρ,p1,p2,q1,q2)

  # precompute Cholesky factor of E
  Einv = cholesky(E)

  # this saves 4 applications of Q that cancel out when calculating the SMW
  # formula
  Â =  linmap_blkdiag_hcat(a)
  QA = Â*Q_N # == Q*A
  #B = [ Q'*QA
  #      sqrt(ρ)*R ]
  #
  nu = size(Q,1)
  ny = size(R,1)
  
  # this is where the Sherman-Morrison-Woodbury formula is used
  function mul!(y,x)
   u = (1/ρ)*Minv*(QA'*Q*x[1:nu] + sqrt(ρ)*R'*x[nu .+ (1:ny)]) # = (1/ρ)*Minv*B'*x
   y .= u - (1/ρ)*Minv*QA'*(Einv\(QA*u))
  end

  mulh!(y,x) = @error("not implemented")

  return LinearMap{ComplexF64}(mul!,mulh!,N*n1*n2,4N*n1*n2,ismutating=true)
 end

 shrink(b,μ) = sign(b)*max(abs(b)-μ,0) # soft thresholding
 """
  Use ADMM to solve

   minimize || A u - d ||_2^2 + ||R u||_1

  Inputs:
    A     system matrix
    R     regularization matrix (includes reg. parameter)
    αs    regularization parameters 
    resmap residual map. y[resmap[i]] gives the indices associated with αs[i]
    v     data / right handside
    u0    initial guess for u
    pinvB least square solver for augmented B = [A; sqrt(ρ)*R]
    ρ      augmented Lagrangian parameter, the larger the closer z tracks R*u
    maxiter (default 10) max number of iterations
    verbose (defaults false) whether to collect convergence stats
    check_convergence function (u,z,y) -> Boolean that indicates whether
                      we should stop iterations. the default behavior is
                      to run for max number of iterations maxiter

  Outputs: (named tuple)
    u         solution u
    z         tracks R*u
    y         Lagrange multipliers
    obj       objective function record
    residuals residual record
 """
 function admm_solver(A,v,u0,ρ,R,resmap,αs,pinvB;maxiter=10,verbose=false,check_convergence=(u,z,y)->false)
  objfun(u,z) = norm(A*u-v)^2/2 + norm(z,1)
  residual(u,z) = norm(R*u-z)^2
  auglag(u,z,y) = norm(A*u-v)^2/2 + norm(z,1) + (ρ/2)*norm(R*u-z)^2 + y'*(R*u-z)

  u = u0; z = 0*R*u; y = 0*z
  #B = [ A
  #      ρ^(1/2)*R]
  obj = residuals = auglags = []
  verbose && (obj       = [ objfun(u,z) ])
  verbose && (residuals = [ residual(u,z)])
  verbose && (auglags = [ auglag(u,z,y)])

  #println("at 0 ",auglag(u,z,y)," ",objfun(u,z)," ",residual(u,z))
  for j=1:maxiter
      # solve augmented least squares problem
      b = [v
           ρ^(1/2)*(z - y/ρ)]
      u .= pinvB*b
      u = (A'*A + ρ*R)\(A'*v + ρ*z - y)

      # solve L2 regularized L1 problem
      ru = R*u
      for (iα,α) ∈ enumerate(αs)
       z[resmap[iα]] .= shrink.(ru[resmap[iα]]+y[resmap[iα]]/ρ,α/ρ)
      end
      #println("shk ",auglag(u,z,y)," ",objfun(u,z)," ",residual(u,z))

      # dual update
      y .+= ρ*(ru - z)
      verbose && println("auglag = ",auglag(u,z,y)," objfun = ",objfun(u,z)," res = ",residual(u,z))
      verbose && push!(obj,objfun(u,z))
      verbose && push!(residuals,residual(u,z))
      verbose && push!(auglags,auglag(u,z,y))
      check_convergence(u,z,y) && break
  end
  return (u=u,z=z,y=y,obj=obj,residuals=residuals,auglags=auglags)
 end
 """
  takes a kernel k and ADMM parameter ρ and does preliminary calculations
  that can be shared between several ADMM calls with same ρ
 """
 function prep_reconstruction(k,p1,q1,p2,q2,N,ρ)
  kn1,kn2,kp1,kp2,kN = size(psf)
  @assert kn1 % p1 == 0 && kn2 % p2 == 0
  @assert kp1 == p1 && kp2 == p2
  @assert kN == N
  n1 = p1*q1; n2 = p2*q2
  @assert kn1 <= n1 && kn2 <= n2

  # zero pad PSF
  k_ = zeros(n1,n2,p1,p2,N)
  k_[1:kn1,1:kn2,:,:,:] = k

  # prepare block matrix representation of PSF
  a = [ pcc2d_to_BlockDiagonal(k_[:,:,:,:,iw]) for iw=1:N]
  A = linmap_blkdiag_hcat(a)

  # regularization matrix
  R = linmap_R(n1,n2,N)

  # indices corresponding to each regularization parameter
  #
  #   α penalizes TV in space, 
  #   β penalizes TV in wavelength, 
  #   γ penalizes l1 norm
  #
  # order: α, α, β, γ
  resmap = residual_indices(n1,n2,N) 

  # prepare augmented least squares solver
  pinvB = linmap_augmented_ls_solver(p1,p2,q1,q2,a,ρ)

  return (A=A,R=R,pinvB=pinvB,ρ=ρ,resmap=resmap,
          p1=p1,q1=q1,p2=p2,q2=q2,N=N)
 end

 function do_reconstruction(b,prep,α,β,γ,x0=[];kwargs...)
  @assert length(b) == prep.n1*prep.n2
  # initial guess
  if isempty(x0)
   u0 = zeros(prep.n1*prep.n2*prep.N)
  else
   @assert length(x0) == prep.n1*prep.n2*prep.N
   u0 = vec(x0)
  end
  αs = [α,α,β,γ]
  admm_solver(prep.A,b,u0,prep.ρ,prep.R,prep.resmap,αs,prep.pinvB;kwargs...)
 end

end#module
