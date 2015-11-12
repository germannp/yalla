// Parallelization for N-Body problem with interactions among all pairs,
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
#include <assert.h>
#include <thrust/extrema.h>

#include "util.cuh"


const uint TILE_SIZE = 32;
__device__ __managed__ float max_F;

extern const float R_MAX;
extern __device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j);


// Calculate new X one thread per cell, to TILE_SIZE other bodies at a time
__global__ void calculate_F(int N_CELLS, float3 X[], float3 F[]) {
    __shared__ float3 shX[TILE_SIZE];
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    float3 Xi = X[cell_idx];
    float3 Fi = {0.0f, 0.0f, 0.0f};
    for (int tile_start = 0; tile_start < N_CELLS; tile_start += TILE_SIZE) {
        int other_cell_idx = tile_start + threadIdx.x;
        shX[threadIdx.x] = X[other_cell_idx];
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            float3 dF = cell_cell_interaction(Xi, shX[i], cell_idx, other_cell_idx);
            Fi.x += dF.x;
            Fi.y += dF.y;
            Fi.z += dF.z;
        }
    }
    F[cell_idx].x = Fi.x;
    F[cell_idx].y = Fi.y;
    F[cell_idx].z = Fi.z;
    atomicMax(&max_F, max(max(Fi.x, Fi.y), Fi.z));
}

__global__ void integrate(float delta_t, int N_CELLS, float3 X[], float3 F[]) {
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell_idx < N_CELLS) {
        X[cell_idx].x += F[cell_idx].x*delta_t;
        X[cell_idx].y += F[cell_idx].y*delta_t;
        X[cell_idx].z += F[cell_idx].z*delta_t;
    }
}


void dynamic_step(float delta_t, int N_CELLS, float3 X[], float3 F[]) {
    int n_blocks = (N_CELLS + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.
    float left_t = delta_t;
    while (left_t > 0) {
        max_F = 0;
        calculate_F<<<n_blocks, TILE_SIZE>>>(N_CELLS, X, F);
        cudaDeviceSynchronize();
        float dt = min(left_t, R_MAX/4/max_F);
        left_t -= dt;
        integrate<<<n_blocks, TILE_SIZE>>>(dt, N_CELLS, X, F);
        cudaDeviceSynchronize();
    }
}
