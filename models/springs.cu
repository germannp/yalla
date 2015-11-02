// Integrate N-body problem with springs between all bodies. Parallelization
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "../lib/vtk.cu"

#define N_CELLS 800
#define TILE_SIZE 16
#define N_TIME_STEPS 100


__device__ __managed__ float3 X[N_CELLS];

__device__ float3 body_body_force(float3 Xi, float3 Xj) {
    float3 r;
    float3 dF = {0.0f, 0.0f, 0.0f};
    r.x = Xj.x - Xi.x;
    r.y = Xj.y - Xi.y;
    r.z = Xj.z - Xi.z;
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > 1e-8) {
        dF.x += r.x*(dist - 0.5)/dist;
        dF.y += r.y*(dist - 0.5)/dist;
        dF.z += r.z*(dist - 0.5)/dist;
    }
    return dF;
}

// Calculate new X one thread per cell, to TILE_SIZE other cells at a time
__global__ void integrate_step() {
    __shared__ float3 shX[TILE_SIZE];
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    float3 Xi = X[cell_idx];
    float3 Fi = {0.0f, 0.0f, 0.0f};
    for (int tile_start = 0; tile_start < N_CELLS; tile_start += TILE_SIZE) {
        int other_cell_idx = tile_start + threadIdx.x;
        shX[threadIdx.x] = X[other_cell_idx];
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            float3 dF = body_body_force(Xi, shX[i]);
            Fi.x += dF.x;
            Fi.y += dF.y;
            Fi.z += dF.z;
        }
    }
    X[cell_idx].x = Xi.x + Fi.x*0.001;
    X[cell_idx].y = Xi.y + Fi.y*0.001;
    X[cell_idx].z = Xi.z + Fi.z*0.001;
}

int main(int argc, const char* argv[]) {
    assert(N_CELLS % TILE_SIZE == 0);

    // Prepare initial state
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

    // Integrate cell positions
    mkdir("output", 755);
    int n_blocks = (N_CELLS + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        std::stringstream file_name;
        file_name << "output/springs_" << time_step << ".vtk";
        write_positions(file_name.str().c_str(), N_CELLS, X);

        if (time_step < N_TIME_STEPS) {
            integrate_step<<<n_blocks, TILE_SIZE>>>();
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
