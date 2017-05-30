The code in this directory has been built with the purpose to generate initial
conditions (as a specific spatial distribution of cells) for the bolls framework
taking 3D mesh models (stl file format) as pre-pattern.

Description of different files.

ic_from_meix.cu : Main program. Takes a 3D mesh file and creates a bolls initial
condition by filling all the volume inside the 3D mesh with bolls. Then writes
the initial conditions in an output file (read comments within main program
on how to execute).

meix.h, meix.cpp : Includes data structures and functions in order to read the
mesh .stl files, store the data and integrate in in the bolls framework.

strtk.hpp : String Tool-kit Library. It's used by meix.cpp to read and manage
the .stl files. I have not written this code. Authorship and copyrights are
indicated at the beginning of the file.

meix_inclusion_test.h : Includes data structures and algorithms to fill the 3D
mesh with bolls (e.g. shooting rays out of bolls and check how many mesh facets
you hit), more details in the code comments.

How to compile (CUDA needed):

nvcc -std=c++11 meix.cpp ic_from_meix.cu
