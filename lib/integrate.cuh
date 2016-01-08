// Kernels to perform Euler steps

__global__ void reset_dX(int n_cells, float3 dX[]) {
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell_idx < n_cells) {
        dX[cell_idx].x = 0;
        dX[cell_idx].y = 0;
        dX[cell_idx].z = 0;
    }
}

__global__ void integrate(int n_cells, float delta_t, float3 X[], float3 dX[]) {
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell_idx < n_cells) {
        X[cell_idx].x += dX[cell_idx].x*delta_t;
        X[cell_idx].y += dX[cell_idx].y*delta_t;
        X[cell_idx].z += dX[cell_idx].z*delta_t;
    }
}
