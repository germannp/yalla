// Simulating cell sorting with limited interactions. Sorting based lattice after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>
#include <thrust/sort.h>

#include "../lib/vtk.cu"


const float R_MAX = 1;
const float R_MIN = 0.5;
const int N_CELLS = 10000;
const int N_TIME_STEPS = 300;
const int GRID_SIZE = 100;
const int N_CUBES = GRID_SIZE*GRID_SIZE*GRID_SIZE;

__device__ __managed__ float3 X[N_CELLS];

__device__ __managed__ int cube_id[N_CELLS];
__device__ __managed__ int cell_id[N_CELLS];
__device__ __managed__ int cube_start[GRID_SIZE*GRID_SIZE*GRID_SIZE];
__device__ __managed__ int cube_end[GRID_SIZE*GRID_SIZE*GRID_SIZE];


__device__ float3 cell_cell_force(float3 Xi, float3 Xj) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (dist > 1e-8) {
        float F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
        dF.x += r.x*F/dist;
        dF.y += r.y*F/dist;
        dF.z += r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__global__ void integrate_step() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        int interacting_cubes[27];
        interacting_cubes[0] = cube_id[i] - 1;
        interacting_cubes[1] = cube_id[i];
        interacting_cubes[2] = cube_id[i] + 1;
        for (int j = 0; j < 3; j++) {
            interacting_cubes[j + 3] = interacting_cubes[j % 3] - GRID_SIZE;
            interacting_cubes[j + 6] = interacting_cubes[j % 3] + GRID_SIZE;
        }
        for (int j = 0; j < 9; j++) {
            interacting_cubes[j +  9] = interacting_cubes[j % 9] - GRID_SIZE*GRID_SIZE;
            interacting_cubes[j + 18] = interacting_cubes[j % 9] + GRID_SIZE*GRID_SIZE;
        }

        float3 dF, F  = {0.0f, 0.0f, 0.0f};
        float3 Xi = X[cell_id[i]];
        for (int j = 0; j < 27; j++) {
            int cube = interacting_cubes[j];
            for (int k = cube_start[cube]; k <= cube_end[cube]; k++) {
                float3 Xj = X[cell_id[k]];
                int strength =
                    (1 + 2*(cell_id[k] < N_CELLS/2))*(1 + 2*(cell_id[i] < N_CELLS/2));
                dF = cell_cell_force(Xi, Xj);
                F.x += strength*dF.x;
                F.y += strength*dF.y;
                F.z += strength*dF.z;
            }
        }
        X[cell_id[i]].x = Xi.x + F.x*0.01;
        X[cell_id[i]].y = Xi.y + F.y*0.01;
        X[cell_id[i]].z = Xi.z + F.z*0.01;
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
        cube_end[i] = -2;
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
    int cell_type[N_CELLS];
    float r_sphere = pow(N_CELLS/0.75, 1./3)*R_MIN/2; // Sphere packing
    for (int i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
        float r = r_sphere*rand()/(RAND_MAX + 1.);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }

    // Integrate cell positions
    mkdir("output", 755);
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        std::stringstream file_name;
        file_name << "output/sorting_" << time_step << ".vtk";
        write_positions(file_name.str().c_str(), N_CELLS, X);
        write_scalars(file_name.str().c_str(), N_CELLS, "cell_type", cell_type);

        if (time_step < N_TIME_STEPS) {
            compute_cube_ids<<<(N_CELLS + 16 - 1)/16, 16>>>();
            cudaDeviceSynchronize();
            thrust::sort_by_key(cube_id, cube_id + N_CELLS, cell_id);
            reset_cubes<<<(N_CUBES + 16 - 1)/16, 16>>>();
            cudaDeviceSynchronize();
            compute_cubes<<<(N_CELLS + 16 - 1)/16, 16>>>();
            cudaDeviceSynchronize();

            integrate_step<<<(N_CELLS + 16 - 1)/16, 16>>>();
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
