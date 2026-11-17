using BenchmarkTools, DataFrames, PyPlot, DiffractogramImaging
DI = DiffractogramImaging

function bench_vary_n()
 results = DataFrame(n=Any[],algo=Any[],bench=Any[])
 p  = (8,8); N = 5; ρ  = 1
 for q1 ∈ 2 .^ (1:7)
   q = (q1,q1)
   n = p.*q 
   k = randn(n...,p...,N)

   # prepare block matrix representation of PSF
   println("tic")
   a = [ DI.pcc2d_to_BlockDiagonal(k[:,:,:,:,iw]) for iw=1:N]
   Q  = DI.linmap_Q(ComplexF64,p...,q...)
   QN = DI.linmap_Q_N(ComplexF64,p...,q...,N)
   A = Q'*DI.linmap_blkdiag_hcat(a)*QN;
   pinvB = DI.linmap_augmented_ls_solver(p...,q...,a,ρ)
   println("toc size=",size(pinvB))
   
   t=@benchmark y=$(A'*A)*$(randn(prod(n)*N)) samples=20
   push!(results,(n,"lsqr",t))
   t=@benchmark y=$(pinvB)*$(randn(4N*prod(n))) samples=20
   push!(results,(n,"pinvB",t))
 end
 return results
end

function bench_vary_nw()
 results = DataFrame(N=Any[],algo=Any[],bench=Any[])
 p  = (8,8); ρ  = 1; q = (4,4);
 n = p.*q
 for N ∈ 2:4:42
   k = randn(n...,p...,N)

   # prepare block matrix representation of PSF
   println("tic")
   a = [ DI.pcc2d_to_BlockDiagonal(k[:,:,:,:,iw]) for iw=1:N]
   Q  = DI.linmap_Q(ComplexF64,p...,q...)
   QN = DI.linmap_Q_N(ComplexF64,p...,q...,N)
   A = Q'*DI.linmap_blkdiag_hcat(a)*QN;
   pinvB = DI.linmap_augmented_ls_solver(p...,q...,a,ρ)
   println("toc size=",size(pinvB))
   
   t=@benchmark y=$(A'*A)*$(randn(prod(n)*N)) samples=20
   push!(results,(N,"lsqr",t))
   t=@benchmark y=$(pinvB)*$(randn(4N*prod(n))) samples=20
   push!(results,(N,"pinvB",t))
 end
 return results
end

# to put any savefig options (here generating PGF and PDF)
function mysavefig(x) 
 savefig("$x.pgf", bbox_inches="tight")
 savefig("$x.pdf", bbox_inches="tight")
end

algos = [ "lsqr", "pinvB"] 
algo_legend = Dict( "lsqr" => "LSQR iter.", "pinvB"=>L"$B^\dagger$" )
figsize = (4,3) .* 0.5

# extract time (in nanoseconds) and memory usage (in bytes)
times(z) = [ mean(z[:,:bench][j].times) for j ∈ 1:size(z,1) ]

# extract time ratios for two algos
function ratios(z,alg1,alg2)
 z1 = z[z.algo .== alg1,:]
 z2 = z[z.algo .== alg2,:]
 return times(z1)./times(z2)
end

function plot_bench_vary_n(res)
  figure(1,figsize=figsize); clf()
  z = res[res.algo .== "lsqr",:]
  ns = prod.(z[:,:n])
  for algo ∈ algos
    z = res[res.algo .== algo,:]
    loglog(ns, times(z)/1e6, label=algo_legend[algo])
  end
  grid(true)
  xlabel(L"|n|")
  ylabel("time (ms)")
  mysavefig("bench_vary_n_times")

  figure(2,figsize=figsize); clf()
  semilogx(ns, ratios(res,"pinvB","lsqr"))
  grid(true)
  xlabel(L"|n|")
  ylabel("time ratio")
  mysavefig("bench_vary_n_ratio")
end

function plot_bench_vary_nw(res)
  figure(1,figsize=figsize); clf()
  z = res[res.algo .== "lsqr",:]
  nws = z[:,:N]
  for algo ∈ algos
    z = res[res.algo .== algo,:]
    loglog(nws, times(z)/1e6, label=algo_legend[algo])
  end
  grid(true)
  xlabel(L"N")
  ylabel("time (ms)")
  mysavefig("bench_vary_nw_times")

  figure(2,figsize=figsize); clf()
  plot(nws, ratios(res,"pinvB","lsqr"))
  grid(true)
  xlabel(L"N")
  ylabel("time ratio")
  mysavefig("bench_vary_nw_ratio")
end

function bench_all()
 res1=bench_vary_n()
 res2=bench_vary_nw()
 return (res1,res2)
end

function plot_all(res1,res2)
 plot_bench_vary_n(res1)
 plot_bench_vary_nw(res2)
end
