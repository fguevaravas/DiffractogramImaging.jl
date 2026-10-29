# # Reconstructions from Macbeth colorchart
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra,BlockDiagonals,Glob
using Profile, ProfileSVG, Printf
DI = DiffractogramImaging;
include("HSPlot.jl");

# ## Set location of data
root = joinpath(homedir(),"lumosdata/realdata")
vpsf = MAT.matread(joinpath(root,"psf_10152019_qev_rnrm.mat"))
vref = MAT.matread(joinpath(root,"refspectra_440-800.mat"))
img_files = glob("mb_*.mat",root)
imgs = Dict()
for f ∈ img_files
    dir,fname = splitdir(f)
    name,ext  = splitext(fname)
    imgs[name] = MAT.matread(f)["X"]
end

# ## Process PSF 
# - remove 425nm wavelength (causes problems)
# - make sure each acquired PSF is rescaled to sum to one
wsidx = 2:26
ws = copy(vpsf["ws"]);
ws = ws[wsidx] 
nw = length(ws)
psf = deepcopy(vpsf["psf"]);
psf = psf[:,:,:,:,wsidx]
for i3=1:size(psf,3),i4=1:size(psf,4),i5=1:size(psf,5)
    s = sum(psf[:,:,i3,i4,i5])
    psf[:,:,i3,i4,i5] /= s
end

# ## Prepare system matrix
# zero pad PSF (all images are 160x160)
q = (20,20); p=(8,8); n = p.*q; N=size(psf,5)
k = DI.psf_zeropad(psf,q...)
## prepare block matrix representation of PSF
a = [ DI.pcc2d_to_BlockDiagonal(k[:,:,:,:,iw]) for iw=1:N]
Q  = DI.linmap_Q(ComplexF64,p...,q...)
QN = DI.linmap_Q_N(ComplexF64,p...,q...,N)
A = Q'*DI.linmap_blkdiag_hcat(a)*QN;


# ## Reconstruction procedure
ρ,α,β,γ = 3e-2,1,1e-8,1e-9 # pretty good
x0 = zeros(prod(p.*q)*N)
prep = DI.prep_reconstruction(k,p...,q...,N,ρ)
reconstruct(x) = DI.do_reconstruction(vec(x),prep,α,β,γ,x0,maxiter=10,verbose=false);

# ## Do the actual reconstructions
xrecs = Dict()
print("reconstructing: ")
for name ∈ keys(imgs)
    xrecs[name]=copy(reshape(reconstruct(imgs[name]).u,n...,N)); 
    print(".")
end
println()

# ## Plot 
cnt=1;
subplots_adjust(wspace=0,hspace=-0.35)
for i ∈ 1:19
    name = "mb_$(i)_rnrm"
    xrgb = HSPlot.hsi2rgb(xrecs[name],ws=ws);
    subplot(4,6,cnt); cnt+=1
    imshow(xrgb); axis("equal"); axis("off")
end

for i ∈ 1:19
    name = "mb_$(i)_rnrm"
    xrgb = HSPlot.hsi2rgb(xrecs[name],ws=ws);
    imsave("mb_$(i)_rnrm.png",xrgb)
end

# ## Sample spectra
# here we sample the spectra near the middle of the image, plot the
# spectra vs the reference one and calculate errors
sample(X,i=(80,80),p=(8,8)) = 
   sum(X[i[1].+(1:p[1]),i[2].+(1:p[2]),:],dims=(1,2))[:]/prod(p)
renorm(x) = x / norm(x,2)
wsref = vref["ws"]
sres = zeros(19)

## For saving
figure(figsize=(3,2)./2)
for i ∈ 1:19
    name = "mb_$(i)_rnrm"
    rec = renorm(sample(xrecs[name]))
    ref = renorm(vref["mb_$i"])
    sres[i] = HSPlot.sre_l2(ws,rec,wsref,ref)
    plot(ws,rec)
    plot(wsref,ref)
    xticks([440,620,800]); 
    yticks([])
    savefig("$(name)_spec.png",bbox_inches="tight",dpi=200)
    clf()
end

## for displaying
figure()
cnt=1;
for i ∈ 1:19
    name = "mb_$(i)_rnrm"
    subplot(4,6,cnt); cnt+=1
    rec = renorm(sample(xrecs[name]))
    ref = renorm(vref["mb_$i"])
    sres[i] = HSPlot.sre_l2(ws,rec,wsref,ref)
    plot(ws,rec)
    plot(wsref,ref)
    axis("off")
end
sres=sres[1:end-1]
println("mean SRE = ",sum(sres)/length(sres),"%")
for i=1:length(sres)
@printf("%.1f%%\n",sres[i])
end
