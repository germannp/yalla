// Parallelization for N-Body problem with interactions among all pairs,
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
#include <assert.h>


const uint TILE_SIZE = 32;

extern __device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j);


// Calculate new X one thread per cell, to TILE_SIZE other bodies at a time
__global__ void integrate(float delta_t, int N_CELLS, float3 X[]) {
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
    X[cell_idx].x = Xi.x + Fi.x*delta_t;
    X[cell_idx].y = Xi.y + Fi.y*delta_t;
    X[cell_idx].z = Xi.z + Fi.z*delta_t;
}

void euler_step(float delta_t, int N_CELLS, float3 X[]) {
    int n_blocks = (N_CELLS + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.
    integrate<<<n_blocks, TILE_SIZE>>>(delta_t, N_CELLS, X);
    cudaDeviceSynchronize();
}
