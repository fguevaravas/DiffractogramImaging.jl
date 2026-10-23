# # Test solver with a synthetic case
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra,BlockDiagonals
using Profile, ProfileSVG
DI = DiffractogramImaging;
include("HSPlot.jl");

# ## A random PSF 
"""
Generate a spatially varying periodic random psf of size p.*q and period p.
The PSF has support sz and `nw` wavelengths. 
Returns a 5D array of size `(q1*p1,q2*p2,p1,p2,nw)`.
"""
function rand_psf(p,q,sz,nw,origin=(0,0))
    n = q.*p
    psf = zeros(n...,p...,nw)
    offset = zeros(Int,2)
    for iw=1:nw, i1=1:p[1], i2=1:p[2]
        offset = origin .+ (i1-1,i2-1) .- sz.÷2
        z = rand(sz...); z = z/sum(z)
        psf[mod1.(offset[1] .+ (1:sz[1]),n[1]),
            mod1.(offset[2] .+ (1:sz[2]),n[2]),
            i1,i2,iw] = z;
    end
    psf
end

kq = (5,5); kp = (3,3); kn=kp.*kq; ksupport = (5,5); N=5; origin = (5,5);
ws = [450,500,550,600,650]
psf = rand_psf(kp,kq,ksupport,N,origin);
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
