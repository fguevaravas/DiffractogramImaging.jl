"""
Tools for visualizing and manipulating hyperspectral images and for
creating simple HTML reports that summarize the data

Note: loadhsdb was not imported and will be put in another package
that handles hyperspectral image databases
"""
module HSPlot

using PyPlot, LinearAlgebra, MAT
import Dierckx
using Interpolations

# bayer_simple, lin2012xyz2e, sbrgb10w, xyz2rgb
include("colorcurves.jl")

"""
     hsi2rgb(hs;mode="xyz",gamma=false,ws=[])

collapses hyperspectral image to RGB

# Inputs
  * `hs`: `Nx x Ny x Nw` hyperspectral image
  * mode
    - `'xyz'`   use XYZ color matching curves and then transform to RGB (**default**)
    - `'rgb'`   use RGB curves directly
    - `'bayer'` uses flat Bayer like filter


  * `gamma`: whether to apply simple gamma correction (default: `false`)
  * `ws`: wavelenghts (defaults to `range(430,stop=718,length=Nw)` if `ws` is empty)

# Outputs
  * `I`: `Nx x Ny x 3` RGB image (with values between 0 and 1)

# Example
```julia
using HSPlot, PyPlot
ws = range(430,stop=718,length=9)
X = loadhsdb(manchester2002,"scene5",sz=(128,128),ws=ws);
imshow(hsi2rgb(X,ws=ws))
```
"""
function hsi2rgb(hs;mode="xyz",gamma=true,ws=[],rescale=true)

    Nx,Ny,Nw = size(hs);
    if (isempty(ws))
        w=range(430,stop=718,length=Nw); # wavelengths (nm)
    else
        w=ws
        @assert Nw == length(w)
    end

    ####### XYZ method
    if mode == "xyz"
        # interpolate curve, note that we set the extrapolation to zero
	# to make sure that anything out of the band is treated as zero
        wxyz = lin2012xyz2e
        X=Dierckx.Spline1D(wxyz[:,1], wxyz[:,2],bc="zero")
        Y=Dierckx.Spline1D(wxyz[:,1], wxyz[:,3],bc="zero")
        Z=Dierckx.Spline1D(wxyz[:,1], wxyz[:,4],bc="zero")
        iwxyz = [X(w) Y(w) Z(w)]

	# we assumes that the wavelengths are uniformly spaced
	# and that they are ordered.
	dw = length(ws)==1 ? 1 : ws[2]-ws[1]
	iwxyz *=  dw

        Ixyz = fill(0.0,Nx,Ny,3);
        # integrate color matching curves
        rhs = max.(hs,0); # get rid of negative vals
        for j=1:3
            for i=1:Nw
                Ixyz[:,:,j]= Ixyz[:,:,j] + iwxyz[i,j]*rhs[:,:,i]
            end
        end
        # transform to rgb

        I = fill(0.0,Nx,Ny,3);
        for i=1:3
            for j=1:3
                I[:,:,i] = I[:,:,i] + xyz2rgb[i,j]*Ixyz[:,:,j]
            end
        end

        ###### RGB method
    elseif mode == "rgb"
        wrgb = copy(sbrgb10w)
        # rescale rgb curves to have same integral
        s=sum(wrgb,dims=1)
        wrgb[:,2:4]=wrgb[:,2:4]*diagm(0 => 1 ./ s[2:4])
        # interpolate curve
        R=Dierckx.Spline1D(wrgb[:,1],wrgb[:,2])
        G=Dierckx.Spline1D(wrgb[:,1],wrgb[:,3])
        B=Dierckx.Spline1D(wrgb[:,1],wrgb[:,4])
        iwrgb=[R(w) G(w) B(w)]
        I = fill(0.0,Nx,Ny,3);
        # integrate color matching curves
        rhs = max.(hs,0);
        for j=1:3
            for i=1:Nw
                I[:,:,j] += iwrgb[i,j]*rhs[:,:,i]
            end
        end

        ###### Imitate Bayer filter
    elseif mode == "bayer"
        wrgb = bayer_simple
        # interpolate curve
        R=interpolate((wrgb[:,1],),wrgb[:,2],Gridded(Constant()))
        G=interpolate((wrgb[:,1],),wrgb[:,3],Gridded(Constant()))
        B=interpolate((wrgb[:,1],),wrgb[:,4],Gridded(Constant()))
        iwrgb=[R(w) G(w) B(w)]
        I = fill(0.0,Nx,Ny,3);
        # integrate color matching curves
        rhs = max.(hs,0);
        for j=1:3
            for i=1:Nw
                I[:,:,j] += iwrgb[i,j]*rhs[:,:,i]
            end
        end
    else
        error("unknown hs method");
    end

    # rescale image and remove any zeros
    I = max.(I,0)
    if (rescale) I/=maximum(I[:]) end

    # gamma correction
    if (gamma) I=I.^0.4; end;

    return I
end#function

""" 
extract RGB+IR representation
"""
function hsi2rgbir(hs,ws;gamma=false,rescale=true,isir=x -> x>800)
 # calculate RGB
 RGB = hsi2rgb(hs,ws=ws,gamma=false,rescale=false,mode="xyz")

 # lump all the IR channels together
 dw = length(ws)==1 ? 1 : ws[2]-ws[1]
 IR = fill(0.0,size(hs)[1:2]) 
 for i=1:length(ws)
  if isir(ws[i])
   IR += dw*hs[:,:,i]
  end
 end

 if rescale
  m = max(maximum(IR),maximum(RGB))
  RGB /=m
  IR  /=m 
 end

 if gamma
  RGB = RGB.^0.4 
  IR =  IR.^0.4 
 end
 
 return (RGB,IR)
end

function save_rgbir(v,ws,rgbfile,irfile;kwargs...)
 RGB,IR = hsi2rgbir(v,ws,kwargs...,rescale=true)
 imsave(rgbfile,RGB)
 imsave(irfile,IR,vmin=0,vmax=1,cmap="gray")
end

"""
     plot_psf(psf,ws=[])

plot spatially varying periodic PSF using RGB representation

# Inputs
- `psf`: 5D array with PSF data
- `ws`: wavelengths (nm) to be used for RGB representation.
  the default is to use the defaults for [`hsi2rgb`](@ref).

# Example
```julia
using PSFMatvec, HSPlot
psf = rand_psf([5,5],[3,3],7);
plot_psf(psf)
```
"""
function plot_psf(psf,ws=[])
qp1,qp2,p1,p2,nw=size(psf);
psf_rgb = fill(0.0,qp1,qp2,p1,p2,3); q = qp1÷p1
for i1=1:p1,i2=1:p2
 psf_rgb[:,:,i1,i2,:] = hsi2rgb(psf[:,:,i1,i2,:],ws=ws)
end
I = fill(0.0,qp1*p1,qp2*p2,3)
for i1=0:(p1-1), i2=0:(p2-1)
 I[i1*qp1 .+ (1:qp1),i2*qp2 .+ (1:qp2),:] = psf_rgb[:,:,i1+1,i2+1,:]
end
I = I/maximum(I[:])
imshow(I,interpolation="none")
for i1=1:p1, i2=1:p2
 annotate("($i1,$i2)",xy=((i1-1)*qp1 + (q-1)/2*p1 + i1,
                       (i2-1)*qp2 + (q-1)/2*p2 + i2),
          color="white");
end
end#function plot_psf


"""
     plothsi(v,s::Tuple{Int,Int})

 Plot hyperspectral image

# Inputs
- `v`: `Nx x Ny x Nw` hyperspectral image
- `s`:  is the size of the matrix with images, column by column ordering
      if `s` is a scalar the default size is `[1,s]`

# Example
```julia
   using PSFMatvec, HSPlot
   psf = rand_psf([5,5],[3,3],7);
   plothsi(psf[:,:,1,1,:],(4,3))
```
"""
function plothsi(v,s::Tuple{Int,Int}; fontsize=6, fontcolor="w", voffset=0)
    Nx,Ny,Nw = size(v)
    if (Nw > s[1]*s[2]) error("s is not big enough"); end;

    # big image to use
    I = fill(0.0,Nx*s[1],Ny*s[2])
    for iw = 1:Nw
        i1,i2 = Tuple(CartesianIndices(s)[iw])
        I[(i1-1)*Nx.+(1:Nx),(i2-1)*Ny.+(1:Ny)] =v[:,:,iw]
    end
    imshow(I,cmap="gray")
    for iw = 1:Nw
        i1,i2 = Tuple(CartesianIndices(s)[iw])
        annotate("$iw", xy = ((i2-1)*Ny+1,voffset+(i1-1)*Nx+1), color=fontcolor, fontsize=fontsize)
    end
end#function plothsi


"""
plot hyperspectral image in rgb separated style

# Inputs
- `v`: `Nx x Ny x Nw` vector of images for each frequency
- `s`: the size of the matrix with images, column  by column ordering
       if `s` is a scalar the default size is `[1,s]`
- `ws`: wavelengths for data set (nm)

# Example
```julia
   using PSFMatvec, HSPlot
   psf = rand_psf([5,5],[3,3],7);
   plothsi_rgbsep(psf[:,:,1,1,:],(4,3))
```
"""
function plothsi_rgbsep(v,s::Tuple{Int,Int},ws=[]; fontsize=6, fontcolor="w", voffset=0)
    Nx,Ny,Nw = size(v)
    if (Nw > s[1]*s[2]) error("s is not big enough"); end;
    z = zero(v)
    m = maximum(v)

    # big image to use
    I = fill(0.0,Nx*s[1],Ny*s[2],3)
    for iw = 1:Nw
        i1,i2 = Tuple(CartesianIndices(s)[iw])
        z[:].=0
        z[:,:,iw] .= max.(v[:,:,iw],0)./m
        I[(i1-1)*Nx.+(1:Nx),(i2-1)*Ny.+(1:Ny),:] .=
            hsi2rgb(z,ws=ws,rescale=false)
    end
    I = I / maximum(I)
    imshow(I)
    for iw = 1:Nw
        i1,i2 = Tuple(CartesianIndices(s)[iw])
        annotate("$iw", xy = ((i2-1)*Ny+1,voffset+(i1-1)*Nx+1),color=fontcolor, fontsize=fontsize)
    end
    return I
end#function plothsi

function save_hsi_rgbsep(v,ws,filefun = x-> "test_out_$(x).png")
    Nx,Ny,Nw = size(v)
    @assert (length(ws) == Nw) "ws and v not consistent"
    m = maximum(v)
    z = zero(v)
    I = fill(0.0,Nx,Ny,3,Nw)
    for iw=1:Nw
     z[:].=0
     z[:,:,iw] .= max.(v[:,:,iw],0)./m
     I[:,:,:,iw] .=hsi2rgb(z,ws=ws,rescale=false)
    end
    I = I / maximum(I)
    for iw=1:Nw
     imsave(filefun(Integer(round(ws[iw]))),I[:,:,:,iw])
    end
end

function save_hsi_sep(v,ws,filefun = x-> "test_out_$(x).png")
    Nx,Ny,Nw = size(v)
    @assert (length(ws) == Nw) "ws and v not consistent"
    I = max.(v,0) / maximum(v)
    for iw=1:Nw
     imsave(filefun(Integer(round(ws[iw]))),I[:,:,iw],
            vmin=0,vmax=1,cmap="gray")
    end
end

function save_hsi_sep(v,ws,s::Tuple{Int,Int},filename::String)
    Nx,Ny,Nw = size(v)
    @assert (Nw <= prod(s)) "s is not big enough"
    @assert (length(ws) == Nw) "ws and v not consistent"

    # big image to use
    I = fill(0.0,Nx*s[1],Ny*s[2])
    for iw = 1:Nw
        i1,i2 = Tuple(CartesianIndices(s)[iw])
        I[(i1-1)*Nx.+(1:Nx),(i2-1)*Ny.+(1:Ny)] =v[:,:,iw]
    end
    I = max.(I,0) / maximum(I)
    imsave(filename,I,vmin=0,vmax=1,cmap="gray")
end

function sre_opt(ws,sp,wsr,spr)
   spri = extrapolate(interpolate((wsr,),spr,Gridded(Linear())),0)
   spref = spri(ws)
   # optimal normalization
   spref /= maximum(spref)
   s = dot(sp,spref)/dot(sp,sp)
   return 100*norm(s*sp - spref)/sqrt(length(ws))
end

end
