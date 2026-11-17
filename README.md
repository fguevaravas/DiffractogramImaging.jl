# Diffractogram imaging

This code accompanies the paper: _Hyperspectral imaging from
diffractograms_ by Fernando González del Cueto, Fernando Guevara
Vasquez, Apratim Majumder and Rajesh Menon. This work was funded by
Lumos Imaging. The code is released under the BSD license.

For the associated preprint see: [arXiv:2611.20544 [math.NA]](https://arxiv.org/abs/2611.20544).

## Testing
```
;cd Diffractogram.jl
]activate .
]test
```

## Generating notebooks
We use `Literate.jl` to generate Jupyter notebooks. The generated
notebooks are not executed, so one still needs to run them (once) to
capture graphical output.
```
;cd Diffractogram.jl/notebooks
]activate .
include("generate_notebooks.jl")
```

List of generated notebooks:
- `test_solver.ipynb`: a very simple test for the solver using a random PSF
- `synthetic_example.ipynb`: a more elaborate example using an acquired PSF and
  a reference hyperspectral image
- `psf_visualization.ipynb`: to generate a visualization of the PSF

### How to create one of the generated notebooks
In order to convert notebook `toconvert.ipynb` to a generated one, we
can start with:
```
jupyter nbconvert toconvert.ipynb --to asciidoc
```
Then follow the simple format in `Literate.jl`.

## Benchmarks
The benchmarks require 30GB just for Julia, with memory usage going higher in the "preparation" phases. They were run on a 32GB machine. To run the benchmarks, in Julia cd to the directory `DiffractogramImaging.jl/benchmark`
```
]activate .
include("run_benchmarks.jl")
res = bench_all()
```

To reproduce the plots:
```
plot_all(res...)
```

The results can be saved using JLD2, which HDF5 based. For example we are using the file `bench.jld2`
```
using JLD2
jldsave("bench.jld2";res)
```

Results can be loaded as follows. The instructions are written assuming this is a new Julia session. First cd to `DiffractogramImaging.jl/benchmark`
```
]activate .
include("run_benchmarks.jl")
using JLD2
res=load("bench.jld2","res")
plot_all(res...)
```

## Funding
This work was partially funded by Lumos Imaging Inc. A. M.
and R. M. acknowledge funding from the Office of Naval Research
(N00014-22-1-2014, N000142512122). F. G. V. acknowledges
funding from the National Science Foundation DMS-2136198.

## License
Unless noted otherwise, the code is released under the following license (see also `LICENSE` file)
```
BSD 3-Clause License

Copyright (c) 2026, Fernando González del Cueto and Fernando Guevara Vasquez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
