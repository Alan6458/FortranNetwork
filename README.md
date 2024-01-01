# FortranNetwork

CrapPyNetwork but it's written in FORTRAN and better

Made for Science Research 2023-2024

## Why you should use FortranNetwork

You probably shouldn't - this was made to further my own understanding of neural networks and FORTRAN. But if you really want to, I won't stop you.

## To compile

Note: Currently, FortranNetwork does not seem to work with gfortran because the gfortran compiler has issues with array allocation. However, FortranNetwork, as of now, works well with the Intel FORTRAN compiler.

### ifx

    ifx [directory]/FortranNetwork.f90 [directory]/[code].f90 -o [directory]/[name]

## To run

    [directory]/[name]
