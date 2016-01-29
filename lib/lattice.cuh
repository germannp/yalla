// Sorting based lattice for N-body problems with limited interaction, after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
#include <assert.h>
#include <thrust/sort.h>

#include "integrutils.cuh"


extern const float R_MAX;
template<typename Pt>
extern __device__ Pt neighbourhood_interaction(Pt Xi, Pt Xj, int i, int j);
template<typename Pt>
extern void global_interactions(const __restrict__ Pt* X, Pt* dX);

const int MAX_n_cells = 1e6;
const int LATTICE_SIZE = 100;
const int N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;
const float CUBE_SIZE = 1;

__device__ __managed__ int cube_id[MAX_n_cells];
__device__ __managed__ int cell_id[MAX_n_cells];
__device__ __managed__ int cube_start[N_CUBES];
__device__ __managed__ int cube_end[N_CUBES];


template<typename Pt>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ X) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        Pt Xi = X[i];
        int id = (int)(
            (floor(Xi.x/CUBE_SIZE) + LATTICE_SIZE/2) +
            (floor(Xi.y/CUBE_SIZE) + LATTICE_SIZE/2)*LATTICE_SIZE +
            (floor(Xi.z/CUBE_SIZE) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
        assert(id >= 0);
        assert(id <= N_CUBES);
        cube_id[i] = id;
        cell_id[i] = i;
    }
}

__global__ void reset_cube_start_and_end() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CUBES) {
        cube_start[i] = -1;
        cube_end[i] = -2;
    }
}

__global__ void compute_cube_start_and_end(int n_cells) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        int cube = cube_id[i];
        int prev = i > 0 ? cube_id[i - 1] : -1;
        if (cube != prev) cube_start[cube] = i;
        int next = i < n_cells ? cube_id[i + 1] : cube_id[i] + 1;
        if (cube != next) cube_end[cube] = i;
    }
}

template<typename Pt>
__global__ void calculate_dX(int n_cells, const Pt* __restrict__ X, Pt* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        int interacting_cubes[27];
        interacting_cubes[0] = cube_id[i] - 1;
        interacting_cubes[1] = cube_id[i];
        interacting_cubes[2] = cube_id[i] + 1;
        for (int j = 0; j < 3; j++) {
            interacting_cubes[j + 3] = interacting_cubes[j % 3] - LATTICE_SIZE;
            interacting_cubes[j + 6] = interacting_cubes[j % 3] + LATTICE_SIZE;
        }
        for (int j = 0; j < 9; j++) {
            interacting_cubes[j +  9] = interacting_cubes[j % 9] - LATTICE_SIZE*LATTICE_SIZE;
            interacting_cubes[j + 18] = interacting_cubes[j % 9] + LATTICE_SIZE*LATTICE_SIZE;
        }

        Pt Fij, F  = zero_Pt();
        Pt Xi = X[cell_id[i]];
        for (int j = 0; j < 27; j++) {
            int cube = interacting_cubes[j];
            for (int k = cube_start[cube]; k <= cube_end[cube]; k++) {
                Pt Xj = X[cell_id[k]];
                Fij = neighbourhood_interaction(Xi, Xj, cell_id[i], cell_id[k]);
                F += Fij;
            }
        }
        dX[cell_id[i]] = F;
    }
}

template<typename Pt>
void heun_step(float delta_t, int n_cells, Pt* X, Pt* dX, Pt* X1, Pt* dX1) {
    assert(LATTICE_SIZE % 2 == 0); // Needed?
    assert(n_cells <= MAX_n_cells);
    assert(R_MAX <= CUBE_SIZE);

    // 1st step
    compute_cube_ids<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X);
    cudaDeviceSynchronize();
    thrust::sort_by_key(cube_id, cube_id + n_cells, cell_id);
    reset_cube_start_and_end<<<(N_CUBES + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();
    compute_cube_start_and_end<<<(n_cells + 16 - 1)/16, 16>>>(n_cells);
    cudaDeviceSynchronize();

    calculate_dX<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X, dX);
    cudaDeviceSynchronize();
    global_interactions(X, dX);
    integrate<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    compute_cube_ids<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X1);
    cudaDeviceSynchronize();
    thrust::sort_by_key(cube_id, cube_id + n_cells, cell_id);
    reset_cube_start_and_end<<<(N_CUBES + 16 - 1)/16, 16>>>();
    cudaDeviceSynchronize();
    compute_cube_start_and_end<<<(n_cells + 16 - 1)/16, 16>>>(n_cells);
    cudaDeviceSynchronize();

    calculate_dX<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X1, dX1);
    cudaDeviceSynchronize();
    global_interactions(X1, dX1);
    integrate<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}
