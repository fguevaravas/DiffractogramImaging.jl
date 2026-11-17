# # PSF Visualization
using Revise, DiffractogramImaging, MAT, LinearMaps, PyPlot, LinearAlgebra,BlockDiagonals,Glob
using Profile, ProfileSVG
DI = DiffractogramImaging;
include("HSPlot.jl");
# ## Load and process PSF
# - remove 425nm wavelength (causes problems)
# - make sure each acquired PSF is rescaled to sum to one
root = joinpath(homedir(),"lumosdata/realdata")
vpsf = MAT.matread(joinpath(root,"psf_10152019_qev_rnrm.mat"))

#wsidx = 2:26
wsidx = 2:4:26
ws = copy(vpsf["ws"]);
ws = ws[wsidx] 
nw = length(ws)
psf = deepcopy(vpsf["psf"]);
psf = psf[:,:,:,:,wsidx]
for i3=1:size(psf,3),i4=1:size(psf,4),i5=1:size(psf,5)
    s = sum(psf[:,:,i3,i4,i5])
    psf[:,:,i3,i4,i5] /= s
end
# crop PSF (note, this cropping is only for visualization purposes only!)
psf = psf[12:31,16:35,:,:,:];
psf = max.(psf,0);
# ## Plotting function
function plot_psf(psf,w)
    qp1,qp2,p1,p2=size(psf);
    q = qp1÷p1
    I = fill(0.0,qp1*p1,qp2*p2,3)
    for i1=0:(p1-1), i2=0:(p2-1)
     thispsf = psf[:,:,i1+1,i2+1]
     white_psf = cat(thispsf,thispsf,thispsf,dims=3).^0.4 # use gamma correction for consistency
     rgb_psf = HSPlot.hsi2rgb(reshape(thispsf,qp1,qp2,1),ws=[w])
     I[i1*qp1 .+ (1:qp1),i2*qp2 .+ (1:qp2),:] = (w<=700) ? rgb_psf : white_psf
    end
    I = I/maximum(I[:])
    imshow(I,interpolation="none")
    for i=1:p1-1
        axhline(y=i*qp1)
        axvline(x=i*qp2)
    end
    #axis("square"); 
    axis("off")
    return I
end;
# ## Plot and save
for i=1:size(psf,5)
    clf()
    I = plot_psf(psf[:,:,:,:,i],ws[i])
    imsave("psf_$(ws[i]).png",I)
end

# ## Extract and save zoomed in values
j = (5,4) # base one index
qp1,qp2,_,_,_ = size(psf)
for i=1:size(psf,5)
    w = ws[i]
    thispsf = psf[:,:,j[1],j[2],i]
    imsave("psf_$(w)_$(j[1])_$(j[2]).png",thispsf.^0.4,cmap="gray")
end
