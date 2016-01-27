// Parallelization for N-Body problem with interactions among all pairs,
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
#include <assert.h>

#include "integrutils.cuh"


const uint TILE_SIZE = 32;

template<typename Pt>
extern __device__ Pt neighbourhood_interaction(Pt Xi, Pt Xj, int i, int j);
template<typename Pt>
extern void global_interactions(const __restrict__ Pt* X, Pt* dX);


// Calculate dX one thread per cell, to TILE_SIZE other bodies at a time
template<typename Pt>
__global__ void calculate_dX(int n_cells, const Pt* __restrict__ X, Pt* dX) {
    __shared__ Pt shX[TILE_SIZE];
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    Pt Xi = X[cell_idx];
    Pt dFi = zero_Pt();

    for (int tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        int other_cell_idx = tile_start + threadIdx.x;
        if (other_cell_idx < n_cells) {
            shX[threadIdx.x] = X[other_cell_idx];
        }
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            int other_cell_idx = tile_start + i;
            if ((cell_idx < n_cells) && (other_cell_idx < n_cells)) {
                Pt Fij = neighbourhood_interaction(Xi, shX[i], cell_idx, other_cell_idx);
                dFi += Fij;
            }
        }
    }

    if (cell_idx < n_cells) {
        dX[cell_idx] += dFi;
    }
}

template<typename Pt>
void heun_step(float delta_t, int n_cells, Pt* X, Pt* dX, Pt* X1, Pt* dX1) {
    int n_blocks = (n_cells + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.

    // 1st step
    reset_dX<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, dX);
    cudaDeviceSynchronize();
    calculate_dX<<<n_blocks, TILE_SIZE>>>(n_cells, X, dX);
    cudaDeviceSynchronize();
    global_interactions(X, dX);
    integrate<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    reset_dX<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, dX1);
    cudaDeviceSynchronize();
    calculate_dX<<<n_blocks, TILE_SIZE>>>(n_cells, X1, dX1);
    cudaDeviceSynchronize();
    global_interactions(X1, dX1);
    integrate<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}
