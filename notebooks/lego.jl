# # A more complicated scene from experimental data
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra,BlockDiagonals,Glob
using Profile, ProfileSVG, TOML
DI = DiffractogramImaging;
include("HSPlot.jl");
default_root = joinpath(homedir(),"lumosdata")
# ## Reconstruction procedure
function reconstruct(psf,file;
        psf_qoffset=(0,0),img_qpad=(0,0),ρ=1e-2,α=1e-1,β=1e-4,γ=1e-6,maxiter=10,
        verbose=false,root = default_root )
    ## load data
    vpsf = MAT.matread(joinpath(root,psf))
    psf = deepcopy(vpsf["psf"]);
    X = MAT.matread(joinpath(root,file))["X"]

    ## check dimensions
    n1,n2=size(X)
    kn1,kn2,p1,p2,N = size(psf)
    p = (p1,p2); n= (n1,n2)
    @assert n1>=kn1 && n2>=kn2
    @assert kn1%p1==0 && kn2%p2==0    

    if "ws" ∈ keys(vpsf)
        ws = copy(vpsf["ws"]);
        N = length(ws)
    else
        ## some old PSF files do not contain the wavelengths
        ws = 425:15:785
        N = length(ws)
    end
    if N == 26
        ## delete 425 band
        wsidx = 2:26
        ws = ws[wsidx] 
        psf = psf[:,:,:,:,2:end]
        N = length(wsidx)
    end
    ## renormalize PSF
    for i3=1:size(psf,3),i4=1:size(psf,4),i5=1:size(psf,5)
        s = sum(psf[:,:,i3,i4,i5])
        psf[:,:,i3,i4,i5] /= s
    end

    ## pad image
    npad = n .+ p.*img_qpad
    Xpad = zeros(npad...)
    Xpad[p1*(img_qpad[1]÷2) .+ (1:n1),
         p2*(img_qpad[2]÷2) .+ (1:n2) ] = X/norm(X,Inf)
    qpad = npad .÷ p

    ## Prepare system matrix
    ## zero pad PSF (all images are 160x160)
    p=(8,8); n = p.*qpad; N=size(psf,5)
    k = DI.psf_zeropad(psf,qpad...,psf_qoffset)
    ## prepare block matrix representation of PSF
    a = [ DI.pcc2d_to_BlockDiagonal(k[:,:,:,:,iw]) for iw=1:N]
    Q  = DI.linmap_Q(ComplexF64,p...,qpad...)
    QN = DI.linmap_Q_N(ComplexF64,p...,qpad...,N)
    A = Q'*DI.linmap_blkdiag_hcat(a)*QN;

    x0 = zeros(prod(p.*qpad)*N)
    prep = DI.prep_reconstruction(k,p...,qpad...,N,ρ)
    res = DI.do_reconstruction(vec(X),prep,α,β,γ,x0,maxiter=maxiter,verbose=verbose);
    return max.(reshape(res.u,npad...,N),0),X,ws
end;

# ## Reconstruction of Lego scene
psf = "psfs/psf_11252018_renorm.mat"
file = "inputs/inputs_12052018/lego_led_onaxis.mat"
opts = (α=2,β=1e-3,ρ=1e-2,maxiter=20)
Xrec,Y,ws=reconstruct(psf,file;opts...)
Xrgb = HSPlot.hsi2rgb(Xrec,ws=ws);
imshow(Xrgb); axis("equal"); axis("off");
imsave("lego_rec.png",Xrgb)
imsave("lego_Y.png",Y,cmap="gray")
