# FortranNetwork

CrapPyNetwork but it's written in FORTRAN and better

Made for Science Research 2023-2024

## Why you should use FortranNetwork

You probably shouldn't - this was made to further my own understanding of neural networks and FORTRAN. But if you really want to, I won't stop you.

## To compile

Note: Although gfortran currently compiles and works with FortranNetwork, it is not recommended to be used because of issues most likely relating to allocatable arrays. Intel's FORTRAN compiler is currently recommended.

### ifx

    ifx [directory]/FortranNetwork.f90 [directory]/[code].f90 -o [directory]/[name]

## To run

    [directory]/[name]
