// Solvers for N-body problem
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "dtypes.cuh"


template<typename Pt>
using nhoodint = Pt (*)(Pt Xi, Pt Xj, int i, int j);

template<typename Pt>
using globints = void (*)(const Pt* __restrict__ X, Pt* dX);

template<typename Pt>
void none(const Pt* __restrict__ X, Pt* dX) {}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution: protected Solver<Pt, N_MAX> {
public:
    __device__ __host__ Pt& operator[](int idx) { return Solver<Pt, N_MAX>::X[idx]; };
    __device__ __host__ const Pt& operator[](int idx) const { return Solver<Pt, N_MAX>::X[idx]; };
    void step(float delta_t, int n_cells, nhoodint<Pt> loc, globints<Pt> glob = none) {
        assert(n_cells <= N_MAX);
        return Solver<Pt, N_MAX>::step(delta_t, n_cells, loc, glob);
    };
};


// Integration templates
template<typename Pt> __global__ void euler_step(int n_cells, float delta_t,
    const Pt* __restrict__ X0, Pt* X, const Pt* __restrict__ dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i] = X0[i] + dX[i]*delta_t;
    }
}

template<typename Pt> __global__ void heun_step(int n_cells, float delta_t,
    const Pt* __restrict__ X0, Pt* X, const Pt* __restrict__ dX,
    const Pt* __restrict__ dX1) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i] = X0[i] + (dX[i] + dX1[i])*0.5*delta_t;
    }
}


/* Parallelization with interactions among all pairs, after
   http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html. */
const uint TILE_SIZE = 32;

template<typename Pt, int N_MAX>class N2nSolver {
protected:
    void step(float delta_t, int n_cells, nhoodint<Pt> local, globints<Pt> global);
    Pt X[N_MAX], dX[N_MAX], X1[N_MAX], dX1[N_MAX];
};

template<typename Pt> __global__ void reset_dX(int n_cells, Pt* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        dX[i] = dX[i]*0;
    }
}

// Calculate dX one thread per cell, to TILE_SIZE other bodies at a time
template<typename Pt>
__global__ void calculate_n2n_dX(int n_cells, const Pt* __restrict__ X, Pt* dX,
    nhoodint<Pt> local) {
    __shared__ Pt shX[TILE_SIZE];
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    Pt Xi = X[cell_idx];
    Pt dFi = Xi*0;

    for (int tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        int other_cell_idx = tile_start + threadIdx.x;
        if (other_cell_idx < n_cells) {
            shX[threadIdx.x] = X[other_cell_idx];
        }
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            int other_cell_idx = tile_start + i;
            if ((cell_idx < n_cells) && (other_cell_idx < n_cells)) {
                Pt Fij = local(Xi, shX[i], cell_idx, other_cell_idx);
                dFi = dFi + Fij;
            }
        }
    }

    if (cell_idx < n_cells) {
        dX[cell_idx] = dX[cell_idx] + dFi;
    }
}

template<typename Pt, int N_MAX>
void N2nSolver<Pt, N_MAX>::step(float delta_t, int n_cells,
    nhoodint<Pt> local, globints<Pt> global) {
    int n_blocks = (n_cells + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.

    // 1st step
    reset_dX<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, dX);
    cudaDeviceSynchronize();
    calculate_n2n_dX<<<n_blocks, TILE_SIZE>>>(n_cells, X, dX, local);
    cudaDeviceSynchronize();
    global(X, dX);
    euler_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    reset_dX<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, dX1);
    cudaDeviceSynchronize();
    calculate_n2n_dX<<<n_blocks, TILE_SIZE>>>(n_cells, X1, dX1, local);
    cudaDeviceSynchronize();
    global(X1, dX1);
    heun_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}


/* Sorting based lattice with limited interaction, after
   http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf */
const float CUBE_SIZE = 1;
const int LATTICE_SIZE = 50;
const int N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<typename Pt, int N_MAX>class LatticeSolver {
protected:
    void step(float delta_t, int n_cells, nhoodint<Pt> local, globints<Pt> global);
    Pt X[N_MAX], dX[N_MAX], X1[N_MAX], dX1[N_MAX];

    int cube_id[N_MAX], cell_id[N_MAX];
    int cube_start[N_CUBES], cube_end[N_CUBES];
};

template<typename Pt>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ X,
    int* cube_id, int* cell_id) {
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

__global__ void reset_cube_start_and_end(int* cube_start, int* cube_end) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CUBES) {
        cube_start[i] = -1;
        cube_end[i] = -2;
    }
}

__global__ void compute_cube_start_and_end(int n_cells, const int* __restrict__ cube_id,
    int* cube_start, int* cube_end) {
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
__global__ void calculate_lattice_dX(int n_cells, const Pt* __restrict__ X, Pt* dX,
    const int* __restrict__ cell_id, const int* __restrict__ cube_id,
    const int* __restrict__ cube_start, const int* __restrict__ cube_end,
    nhoodint<Pt> local) {
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

        Pt Xi = X[cell_id[i]];
        Pt Fij, F = Xi*0;
        for (int j = 0; j < 27; j++) {
            int cube = interacting_cubes[j];
            for (int k = cube_start[cube]; k <= cube_end[cube]; k++) {
                Pt Xj = X[cell_id[k]];
                Fij = local(Xi, Xj, cell_id[i], cell_id[k]);
                F = F + Fij;
            }
        }
        dX[cell_id[i]] = F;
    }
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::step(float delta_t, int n_cells,
    nhoodint<Pt> local, globints<Pt> global) {
    assert(LATTICE_SIZE % 2 == 0); // Needed?

    // 1st step
    compute_cube_ids<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, X, cube_id, cell_id);
    cudaDeviceSynchronize();
    thrust::sort_by_key(thrust::device, cube_id, cube_id + n_cells, cell_id);
    reset_cube_start_and_end<<<(N_CUBES + 32 - 1)/32, 32>>>(cube_start, cube_end);
    cudaDeviceSynchronize();
    compute_cube_start_and_end<<<(n_cells + 32 - 1)/32, 32>>>(n_cells,
        cube_id, cube_start, cube_end);
    cudaDeviceSynchronize();

    calculate_lattice_dX<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X, dX,
        cell_id, cube_id, cube_start, cube_end, local);
    cudaDeviceSynchronize();
    global(X, dX);
    euler_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    compute_cube_ids<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, X1, cube_id, cell_id);
    cudaDeviceSynchronize();
    thrust::sort_by_key(thrust::device, cube_id, cube_id + n_cells, cell_id);
    reset_cube_start_and_end<<<(N_CUBES + 32 - 1)/32, 32>>>(cube_start, cube_end);
    cudaDeviceSynchronize();
    compute_cube_start_and_end<<<(n_cells + 32 - 1)/32, 32>>>(n_cells,
        cube_id, cube_start, cube_end);
    cudaDeviceSynchronize();

    calculate_lattice_dX<<<(n_cells + 16 - 1)/16, 16>>>(n_cells, X1, dX1,
        cell_id, cube_id, cube_start, cube_end, local);
    cudaDeviceSynchronize();
    global(X1, dX1);
    heun_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}
