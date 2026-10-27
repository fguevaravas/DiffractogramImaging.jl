# # Synthetic reconstruction example using acquired PSF
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra
using Interpolations, Random, Distributions
DI = DiffractogramImaging;
include("HSPlot.jl");

# ## Load reference hyperspectral image
# We need an image called `scene5.mat` (a Matlab file) that can be downloaded from [Hyperspectral
# Images of Natural Scenes
# 2002](https://personalpages.manchester.ac.uk/staff/d.h.foster/Hyperspectral_images_of_natural_scenes_02.html).
# The source publication is [Foster, D.H., Amano, K., Nascimento, S.M.C., &
# Foster, M.J. (2006). Frequency of metamerism in natural scenes. Journal of the
# Optical Society of America A, 23,
# 2359-2372](https://doi.org/10.1364/JOSAA.23.002359). Please edit below to reflect
# path to this .mat file.
# 
# The `lazy` variable below works as follows. If true assumes reconstruction has
# already been computed and saved to `synth_Xrec.mat` (in the current working
# directory). If false, recomputes reconstruction and saves to `synth_Xrec.mat`
lazy = true 
file = joinpath(homedir(),"lumosdata/naturalhsi2002/scene5.mat")
v = matread(file);
wsraw = range(410,710,length=31); 
n1raw,n2raw,nwsraw = size(v["reflectances"])
itp = interpolate(((0:n1raw-1)./n1raw,(0:n2raw-1)./n2raw,wsraw), v["reflectances"], Gridded(Linear()));

# ## Undersample image in space and wavelength
# This is done via linear interpolation
norig = (472,472)
ws = 440:30:710
Xorig = itp((0:norig[1]-1)./norig[1],(0:norig[2]-1)./norig[2],ws);


# ## Load experimental PSF
# Here we load the experimental PSF and restrict it to the 10 wavelengths 440:30:710 (nm)
root = joinpath(homedir(),"lumosdata/realdata")
psfname = "psf_10152019_qev_rnrm.mat"
vpsf = MAT.matread(joinpath(root,psfname))
psf = deepcopy(vpsf["psf"]);
ws = copy(vpsf["ws"]);

## check dimensions
kn1,kn2,p1,p2,N = size(psf)
p = (p1,p2);
@assert kn1%p1==0 && kn2%p2==0    
widx = 2:2:20
ws = ws[widx] 
psf = psf[:,:,:,:,widx]
N = length(widx)
## renormalize PSF
for i3=1:size(psf,3),i4=1:size(psf,4),i5=1:size(psf,5)
    s = sum(psf[:,:,i3,i4,i5])
    psf[:,:,i3,i4,i5] /= s
end

# ## Forward problem
# Here we zero pad the input image and the PSF. The we calculate the effect of
# the PSF on the hyperspectral image to obtain a diffractogram `Y`.
qpad = (5,5)
n = norig .+ p.*qpad
q = n .÷ p

k_ = DI.psf_zeropad(psf,q...,(-2,-2))

Xtrue = zeros(n...,N)
Xtrue[p[1]*(qpad[1]÷2) .+ (1:norig[1]),
      p[2]*(qpad[2]÷2) .+ (1:norig[2]),:] .= Xorig

## prepare block matrix representation of PSF
a = [ DI.pcc2d_to_BlockDiagonal(k_[:,:,:,:,iw]) for iw=1:N]
Q  = DI.linmap_Q(ComplexF64,p...,q...)
QN = DI.linmap_Q_N(ComplexF64,p...,q...,N)
A = Q'*DI.linmap_blkdiag_hcat(a)*QN;
Y = real.(reshape(A*vec(Xtrue),n...));

# ## Add noise
# Here we add Poisson noise, pretending we are working with a 12bit sensor (i.e.
# $2^{12} = 4096$ levels of gray). The Poisson process rate is the value at the
# given pixel. Note that the acquired PSF has some noise that may make `Y` have
# small negative values, so we cut these values off.
Random.seed!(17)
Y = max.(Y,0)
Y = 2^12 * Y / norm(Y,Inf);
Ynoisy = rand.(Distributions.Poisson.(Y))
Y = Ynoisy / norm(Ynoisy,Inf);

# ## Plot RGB and diffractogram
Xrgb = HSPlot.hsi2rgb(Xtrue,ws=ws); 
imsave("synth_Xtrue_rgb.png",Xrgb,); 
imsave("synth_Y.png",Y,cmap="gray"); 
HSPlot.save_hsi_rgbsep(Xtrue,ws,x->"synth_Xtrue_sep_$(x).png")

subplot(1,2,1)
imshow(Xrgb); axis("equal"); axis("off")
subplot(1,2,2)
imshow(Y,cmap="gray"); axis("equal"); axis("off");

# ## Reconstructions
if !lazy
  ρ,α,β,γ = 5e-3,5e-1,1e-5,1e-10 
  x0 = 0*vec(Xtrue)
  prep = DI.prep_reconstruction(k_,p...,q...,N,ρ)
  @time res=DI.do_reconstruction(vec(Y),prep,α,β,γ,x0,maxiter=10,verbose=true)
  Xrec = reshape(max.(res.u,0),n...,N);
  MAT.matwrite("synth_Xrec.mat",
         Dict("Xrec"=>Xrec,"auglags"=>res.auglags,"obj"=>res.obj,"residuals"=>res.residuals));
else
  xrf = MAT.matread("synth_Xrec.mat");
  Xrec = xrf["Xrec"]
  res = (obj=xrf["obj"],residuals=xrf["residuals"],auglags=xrf["auglags"])
end;

# ## Plotting
figure()
HSPlot.plothsi(Xrec,(2,5));
axis("off");
HSPlot.save_hsi_rgbsep(Xrec,ws,x->"synth_Xrec_sep_$(x).png")

figure()
semilogy(res.obj,label="objective")
semilogy(res.residuals,label="residual")
semilogy(abs.(res.auglags),label="Lagrangian")
xlabel("iteration")
legend()
savefig("synth_convergence.pgf")


# ## Plot reconstruction with and without spectrum samples
sample_locs = [ (132,242), (195,242), (256,245), (332,341), (218,380) ]
sample(X,i=(80,80),p=(8,8)) = 
   sum(X[i[1].+(1:p[1]),i[2].+(1:p[2]),:],dims=(1,2))[:]/prod(p)
renorm(x) = x / norm(x,1)
Xrec_rgb = HSPlot.hsi2rgb(Xrec,ws=ws);
imsave("synth_Xrec_rgb.png",Xrec_rgb); 
imshow(Xrec_rgb); 
for (i,s) ∈ enumerate(sample_locs)
    text(s[2],s[1],"$i")
end
axis("equal"); axis("off");
savefig("synth_Xrec_rgb_annotated.pgf")

# ## Plot spectrum samples
 sres = zeros(length(sample_locs))
figure(figsize=(3,3))
tight_layout(pad=0)
for (i,s) ∈ enumerate(sample_locs)
    strue=renorm(sample(Xtrue,s))
    srec=renorm(sample(Xrec,s))
    sres[i] = HSPlot.sre_opt(ws,srec,ws,strue)
    clf()
    plot(ws,strue)
    plot(ws,srec)
    ylim(0,0.4)
    axis("off")
    savefig("synth_spec_$(i).pgf")
end

println("sres = ",sres)
println("Average SRE = ",sum(sres)/length(sres))


# ## Zoom regions
zoomin = [ (offset=(65,300), extent=(128,128)),
           (offset=(190,90), extent=(128,128)),
           (offset=(252,340), extent=(128,128))]
zoom(X,z) = X[z.offset[1] .+ (0:z.extent[1]-1),
              z.offset[2] .+ (0:z.extent[2]-1),
              :]
for (iz,z) ∈ enumerate(zoomin)
    Xtrue_zoom = zoom(Xtrue,z)
    Xrec_zoom = zoom(Xrec,z)
    Y_zoom = dropdims(zoom(Y,z),dims=3)
    Xtrue_zoom_rgb = HSPlot.hsi2rgb(Xtrue_zoom,ws=ws);
    Xrec_zoom_rgb = HSPlot.hsi2rgb(Xrec_zoom,ws=ws);
    imsave("synth_Xtrue_zoom$(iz)_rgb.png",Xtrue_zoom_rgb)
    imsave("synth_Xrec_zoom$(iz)_rgb.png",Xrec_zoom_rgb)
    imsave("synth_Y_zoom$(iz).png",Y_zoom,cmap="gray")
end

