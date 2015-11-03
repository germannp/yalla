// Simulating cell sorting with limited interactions. Sorting based grid after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "../lib/vtk.cu"


const float R_MAX = 1;
const int GRID_SIZE = 100;
const int N_CELLS = 800;

__device__ __managed__ float3 X[N_CELLS];
__device__ __managed__ int grid_id[N_CELLS];


__global__ void construct_grid() {
    int cell_idx = blockIdx.x*16 + threadIdx.x;
    float3 Xi = X[cell_idx];
    int id = (int)(
        (floor(Xi.x/R_MAX) + GRID_SIZE/2) +
        (floor(Xi.y/R_MAX) + GRID_SIZE/2)*GRID_SIZE +
        (floor(Xi.z/R_MAX) + GRID_SIZE/2)*GRID_SIZE*GRID_SIZE);
    assert(id >= 0);
    assert(id <= GRID_SIZE*GRID_SIZE*GRID_SIZE);
    grid_id[cell_idx] = id;
    printf("%f %i\n", Xi.x, grid_id[cell_idx]);
}


int main(int argc, char const *argv[]) {
    assert(GRID_SIZE % 2 == 0);

    // Prepare initial state
    float r_sphere = pow(N_CELLS/0.75, 1./3)*R_MAX/2; // Sphere packing
    for (int i = 0; i < N_CELLS; i++) {
        float r = r_sphere*rand()/(RAND_MAX + 1.);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }

    int n_blocks = (N_CELLS + 16 - 1)/16; // ceil int div.
    construct_grid<<<n_blocks, 16>>>();
    cudaDeviceSynchronize();

    std::stringstream file_name;
    file_name << "output/sorting" << ".vtk";
    write_positions(file_name.str().c_str(), N_CELLS, X);
    write_scalars(file_name.str().c_str(), N_CELLS, "grid_id", grid_id);

    return 0;
}
