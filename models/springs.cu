#include <iostream>
#include <sstream>
#include <cmath>

#include "../lib/vtk.cu"

#define N_CELLS 100


__device__ __managed__ float3 X[N_CELLS];

int main(int argc, const char* argv[]) {
    float r_min = 0.5;
    float r_max = pow(N_CELLS/0.75, 1./3)*r_min/2; // Sphere packing
    for (int i = 0; i < N_CELLS; i++) {
        float r = r_max*rand()/(RAND_MAX + 1.);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }

    for (int time_step = 0; time_step < 10; time_step++) {
        std::stringstream file_name;
        file_name << "springs_" << time_step << ".vtk";
        write_positions(file_name.str().c_str(), N_CELLS, X);
    }

    return 0;
}
