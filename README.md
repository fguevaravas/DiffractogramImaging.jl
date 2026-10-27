# Diffractogram imaging

This code accompanies the paper: _Hyperspectral imaging from
diffractograms_ by Fernando González del Cueto, Fernando Guevara
Vasquez, Apratim Majumder and Rajesh Menon. This work was funded by
Lumos Imaging. The code is released under the BSD license.

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
