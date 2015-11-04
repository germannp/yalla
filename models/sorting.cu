// Simulating cell sorting with limited interactions. Sorting based lattice after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <thrust/sort.h>
#include <sys/stat.h>
#include <thrust/sort.h>

#include "../lib/vtk.cu"


const float R_MAX = 1;
const int N_CELLS = 1;
const int GRID_SIZE = 4;
const int N_CUBES = GRID_SIZE*GRID_SIZE*GRID_SIZE;

__device__ __managed__ float3 X[N_CELLS];

__device__ __managed__ int cube_id[N_CELLS];
__device__ __managed__ int cell_id[N_CELLS];
__device__ __managed__ int cube_start[GRID_SIZE*GRID_SIZE*GRID_SIZE];
__device__ __managed__ int cube_end[GRID_SIZE*GRID_SIZE*GRID_SIZE];


__global__ void find_interactions() {
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell_idx < N_CELLS) {
        int cube = cube_id[cell_idx];
        int interacting_cubes[27];
        interacting_cubes[0] = cube - 1;
        interacting_cubes[1] = cube;
        interacting_cubes[2] = cube + 1;
        for (int j = 0; j < 3; j++) {
            interacting_cubes[j + 3] = interacting_cubes[j % 3] - GRID_SIZE;
            interacting_cubes[j + 6] = interacting_cubes[j % 3] + GRID_SIZE;
        }
        for (int j = 0; j < 9; j++) {
            interacting_cubes[j +  9] = interacting_cubes[j % 9] - GRID_SIZE*GRID_SIZE;
            interacting_cubes[j + 18] = interacting_cubes[j % 9] + GRID_SIZE*GRID_SIZE;
        }
        printf("cube_id %i\n", cube);
        for (int j = 0; j < 27; j++) {
            printf("%i\n", interacting_cubes[j]);
        }
    }
}


__global__ void compute_cube_ids() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        float3 Xi = X[i];
        int id = (int)(
            (floor(Xi.x/R_MAX) + GRID_SIZE/2) +
            (floor(Xi.y/R_MAX) + GRID_SIZE/2)*GRID_SIZE +
            (floor(Xi.z/R_MAX) + GRID_SIZE/2)*GRID_SIZE*GRID_SIZE);
        assert(id >= 0);
        assert(id <= GRID_SIZE*GRID_SIZE*GRID_SIZE);
        cube_id[i] = id;
        cell_id[i] = i;
    }
}

__global__ void reset_cubes() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CUBES) {
        cube_start[i] = -1;
        cube_end[i] = -1;
    }
}

__global__ void compute_cubes() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        int cube = cube_id[i];
        int prev = i > 0 ? cube_id[i - 1] : -1;
        if (cube != prev) cube_start[cube] = i;
        int next = i < N_CELLS ? cube_id[i + 1] : cube_id[i] + 1;
        if (cube != next) cube_end[cube] = i;
    }
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

    std::stringstream file_name;
    file_name << "output/sorting" << ".vtk";
    write_positions(file_name.str().c_str(), N_CELLS, X);

    // Update lattice
    compute_cube_ids<<<(N_CELLS + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();
    thrust::sort_by_key(cube_id, cube_id + N_CELLS, cell_id);
    reset_cubes<<<(N_CUBES + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();
    compute_cubes<<<(N_CELLS + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();

    write_scalars(file_name.str().c_str(), N_CELLS, "cube_id", cube_id);

    // Check for neighbors
    find_interactions<<<(N_CELLS + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();

    return 0;
}
