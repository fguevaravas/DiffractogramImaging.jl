# Diffractogram imaging

This code accompanies the paper: _Hyperspectral imaging from diffractograms_ by Fernando González del Cueto, Fernando Guevara Vasquez, Apratim Majumder and Rajesh Menon. This work was funded by Lumos Imaging. The code is released under the BSD license.

## Testing
```
;cd Diffractogram.jl
]activate .
]test
```

## Generating notebooks
Note that the notebooks are not executed, you still need to run them once to capture graphical output.
```
;cd Diffractogram.jl/notebooks
]activate .
include("generate_notebooks.jl")
```
