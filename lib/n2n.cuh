// Parallelization for N-Body problem with interactions among all pairs,
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
#include <assert.h>

#include "integrate.cuh"


const uint TILE_SIZE = 32;

extern __device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j);


// Calculate dX one thread per cell, to TILE_SIZE other bodies at a time
__global__ void calculate_dX(int n_cells, const float3 __restrict__ X[], float3 dX[]) {
    __shared__ float3 shX[TILE_SIZE];
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    float3 Xi = X[cell_idx];
    float3 dFi = {0.0f, 0.0f, 0.0f};

    for (int tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        int other_cell_idx = tile_start + threadIdx.x;
        if (other_cell_idx < n_cells) {
            shX[threadIdx.x] = X[other_cell_idx];
        }
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            int other_cell_idx = tile_start + i;
            if ((cell_idx < n_cells) && (other_cell_idx < n_cells)) {
                float3 Fij = cell_cell_interaction(Xi, shX[i], cell_idx, other_cell_idx);
                dFi.x += Fij.x;
                dFi.y += Fij.y;
                dFi.z += Fij.z;
            }
        }
    }

    if (cell_idx < n_cells) {
        dX[cell_idx].x += dFi.x;
        dX[cell_idx].y += dFi.y;
        dX[cell_idx].z += dFi.z;
    }
}

void euler_step(float delta_t, int n_cells, float3 X[], float3 dX[]) {
    int n_blocks = (n_cells + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.

    reset_dX<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, dX);
    cudaDeviceSynchronize();
    calculate_dX<<<n_blocks, TILE_SIZE>>>(n_cells, X, dX);
    cudaDeviceSynchronize();
    integrate<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, dX);
    cudaDeviceSynchronize();
}
