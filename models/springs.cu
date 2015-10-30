#include <iostream>
#include <cmath>

#include "../lib/vtk.cu"

#define N_CELLS 100


__device__ __managed__ float3 X[N_CELLS];

int main(int argc, char const *argv[]) {
    float r_min = 0.5;
    float r_max = pow(N_CELLS/0.75, 1./3)*r_min/2; /* Sphere packing */
    for (int i = 0; i < N_CELLS; i++) {
        float r = r_max*rand()/(RAND_MAX + 1.);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }
    write_positions("descht.vtk", N_CELLS, X);
    return 0;
}
