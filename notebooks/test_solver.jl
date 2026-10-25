# # Test solver with a synthetic case
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra,BlockDiagonals
using Profile, ProfileSVG
DI = DiffractogramImaging;
include("HSPlot.jl");

# ## A random PSF 
kq = (5,5); kp = (3,3); kn=kp.*kq; ksupport = (5,5); N=5; origin = (5,5);
ws = [450,500,550,600,650]
psf = DI.psf_rand(kp,kq,ksupport,N,origin);
HSPlot.plot_psf(psf,ws)

# ## Generate true signal
p = kp; q = (20,20); n = p.*q
## signal = [0,0,3,0,0]
signal = [1,2,3,2,1]
Xtrue = zeros(n...,N)
for j=1:N
 Xtrue[ n[1]÷2 .+ (1:10),  n[2]÷2 .+ (1:10), j ] .= signal[j];
end
probe(X) = X[5 + (n[1]÷2), 5 + (n[2]÷2), :];

# ## Prepare system matrix
## zero pad PSF
k_ = zeros(n...,p...,N)
k_[1:kn[1],1:kn[2],:,:,:] .= psf

## prepare block matrix representation of PSF
a = [ DI.pcc2d_to_BlockDiagonal(k_[:,:,:,:,iw]) for iw=1:N]
Q  = DI.linmap_Q(ComplexF64,p...,q...)
QN = DI.linmap_Q_N(ComplexF64,p...,q...,N) 
A = Q'*DI.linmap_blkdiag_hcat(a)*QN;
Y = real.(reshape(A*vec(Xtrue),n...));

subplot(1,2,1)
HSPlot.plothsi(Xtrue,(3,3)); axis("equal"); axis("off"); 
subplot(1,2,2)
imshow(Y); axis("equal"); axis("off"); 

#  ## Reconstruct
ρ,α,β,γ = 1e-2,1e-2,1e-6,1e-6
x0 = 0*vec(Xtrue)
prep = DI.prep_reconstruction(psf,p...,q...,N,ρ)
res=DI.do_reconstruction(vec(Y),prep,α,β,γ,x0,maxiter=20,verbose=true);
Xrec = reshape(max.(res.u,0),n...,N)
subplot(2,2,1)
HSPlot.plothsi(Xrec,(3,3)); #colorbar()
subplot(2,2,2)
semilogy(res.obj,label="obj")
semilogy(res.residuals,label="res")
semilogy(abs.(res.auglags),label="lag")
legend()
subplot(2,2,3)
plot(probe(Xrec),label="rec")
plot(probe(Xtrue),label="true")
legend()
