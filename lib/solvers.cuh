// Solvers for N-body problems
#pragma once

#include <assert.h>
#include <functional>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cudebug.cuh"


// A function Pt pairwise_interaction(Pt Xi, Pt Xj, int i, int j) defining the
// interaction between each pair of points of type Pt (e.g. float3, see dtypes.cuh)
// must be defined before #including this file to allow inlining in kernels.

// Optional a generic force with the following signature can be passed:
template<typename Pt>
using GenericForces = std::function<void (const Pt* __restrict__ d_X, Pt* d_dX)>;

template<typename Pt>
void none(const Pt* __restrict__ d_X, Pt* d_dX) {}


// Solution<Pt, N_MAX, Solver> combines a method, Solver, with a point type, Pt.
// It stores the solutions on the host and specifies how the solution on the
// device can be accessed and how new steps are calculated. However all the GPU
// action is happening in the Solver classes.
template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution: public Solver<Pt, N_MAX> {
public:
    Pt *h_X = (Pt*)malloc(N_MAX*sizeof(Pt));  // Current solution on host
    Pt *d_X = Solver<Pt, N_MAX>::d_X;         // Solution on device (GPU)
    int *h_n = (int*)malloc(sizeof(int));     // Number of bolls
    int *d_n = Solver<Pt, N_MAX>::d_n;
    Solution(int n = N_MAX) {
        *h_n = n;
    }
    void memcpyHostToDevice() {
        assert(*h_n <= N_MAX);
        cudaMemcpy(d_X, h_X, N_MAX*sizeof(Pt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_X, d_X, N_MAX*sizeof(Pt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= N_MAX);
    }
    int get_d_n() {
        return Solver<Pt, N_MAX>::get_d_n();
    }
    void step(float delta_t) {
        return Solver<Pt, N_MAX>::step(delta_t, none<Pt>);
    }
    void step(float delta_t, GenericForces<Pt> genforce) {
        return Solver<Pt, N_MAX>::step(delta_t, genforce);
    }
};


// Integration templates
template<typename Pt> __global__ void euler_step(int n_cells, float delta_t,
        const Pt* __restrict__ d_X0, Pt* d_X, const Pt* __restrict__ d_dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    D_ASSERT(d_dX[i].x == d_dX[i].x);  // For NaN f != f
    d_X[i] = d_X0[i] + d_dX[i]*delta_t;
}

template<typename Pt> __global__ void heun_step(int n_cells, float delta_t,
        Pt* d_X, const Pt* __restrict__ d_dX, const Pt* __restrict__ d_dX1) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    D_ASSERT(d_dX1[i].x == d_dX1[i].x);
    d_X[i] += (d_dX[i] + d_dX1[i])*0.5*delta_t;
}


// Solver implementation with interactions among all pairs, after
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, int N_MAX>class N2nSolver {
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    int *d_n;
    N2nSolver() {
        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));

        cudaMalloc(&d_n, sizeof(int));
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= N_MAX);
        return n;
    }
    void step(float delta_t, GenericForces<Pt> genforce);
};

// Calculate d_dX one thread per point, to TILE_SIZE other points at a time
template<typename Pt>
__global__ void calculate_n2n_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX) {
    auto d_cell_idx = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ Pt shX[TILE_SIZE];
    auto Xi = d_X[d_cell_idx];
    Pt Fi {0};

    for (auto tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        auto other_d_cell_idx = tile_start + threadIdx.x;
        if (other_d_cell_idx < n_cells) {
            shX[threadIdx.x] = d_X[other_d_cell_idx];
        }
        __syncthreads();
        for (auto i = 0; i < TILE_SIZE; i++) {
            auto other_d_cell_idx = tile_start + i;
            if ((d_cell_idx < n_cells) and (other_d_cell_idx < n_cells)) {
                Fi += pairwise_interaction(Xi, shX[i], d_cell_idx, other_d_cell_idx);
            }
        }
    }

    if (d_cell_idx < n_cells) {
        d_dX[d_cell_idx] = Fi;
    }
}

template<typename Pt, int N_MAX>
void N2nSolver<Pt, N_MAX>::step(float delta_t, GenericForces<Pt> genforce) {
    auto n = get_d_n();

    // 1st step
    calculate_n2n_dX<<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(  // ceil int div.
        n, d_X, d_dX);
    genforce(d_X, d_dX);
    euler_step<<<(n + 32 - 1)/32, 32>>>(n, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    calculate_n2n_dX<<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(
        n, d_X1, d_dX1);
    genforce(d_X1, d_dX1);
    heun_step<<<(n + 32 - 1)/32, 32>>>(n, delta_t, d_X, d_dX, d_dX1);
}


// Solver implementation with sorting based lattice for limited pairwise_interaction,
// after http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto LATTICE_SIZE = 50u;
const auto N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<int N_MAX>struct Lattice {
public:
    int *d_cube_id, *d_cell_id, *d_cube_start, *d_cube_end;
    Lattice() {
        cudaMalloc(&d_cube_id, N_MAX*sizeof(int));
        cudaMalloc(&d_cell_id, N_MAX*sizeof(int));
        cudaMalloc(&d_cube_start, N_CUBES*sizeof(int));
        cudaMalloc(&d_cube_end, N_CUBES*sizeof(int));
    }
};

__constant__ int d_moore_nhood[27];  // Yes, this is a waste if no LatticeSolver is used

template<typename Pt, int N_MAX>class LatticeSolver {
public:
    Lattice<N_MAX> lattice;
    Lattice<N_MAX> *d_lattice;
    void build_lattice(float cube_size) {
        build_lattice(d_X, cube_size);
    };
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    int *d_n;
    LatticeSolver() {
        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));

        cudaMalloc(&d_lattice, sizeof(Lattice<N_MAX>));
        cudaMemcpy(d_lattice, &lattice, sizeof(Lattice<N_MAX>), cudaMemcpyHostToDevice);

        cudaMalloc(&d_n, sizeof(int));

        int h_moore_nhood[27];
        h_moore_nhood[0] = - 1;
        h_moore_nhood[1] = 0;
        h_moore_nhood[2] = 1;
        for (auto i = 0; i < 3; i++) {
            h_moore_nhood[i + 3] = h_moore_nhood[i % 3] - LATTICE_SIZE;
            h_moore_nhood[i + 6] = h_moore_nhood[i % 3] + LATTICE_SIZE;
        }
        for (auto i = 0; i < 9; i++) {
            h_moore_nhood[i +  9] = h_moore_nhood[i % 9] - LATTICE_SIZE*LATTICE_SIZE;
            h_moore_nhood[i + 18] = h_moore_nhood[i % 9] + LATTICE_SIZE*LATTICE_SIZE;
        }
        cudaMemcpyToSymbol(d_moore_nhood, &h_moore_nhood, 27*sizeof(int));
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= N_MAX);
        return n;
    }
    void step(float delta_t, GenericForces<Pt> genforce);
    void build_lattice(const Pt* __restrict__ d_X, float cube_size = CUBE_SIZE);
};


// Build lattice
template<typename Pt, int N_MAX>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ d_X,
        Lattice<N_MAX>* d_lattice, float cube_size) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x/cube_size) + LATTICE_SIZE/2) +
        (floor(d_X[i].y/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE +
        (floor(d_X[i].z/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
    D_ASSERT(id >= 0);
    D_ASSERT(id <= N_CUBES);
    d_lattice->d_cube_id[i] = id;
    d_lattice->d_cell_id[i] = i;
}

template<int N_MAX>
__global__ void compute_cube_start_and_end(int n_cells, Lattice<N_MAX>* d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto cube = d_lattice->d_cube_id[i];
    auto prev = i > 0 ? d_lattice->d_cube_id[i - 1] : -1;
    if (cube != prev) d_lattice->d_cube_start[cube] = i;
    auto next = i < n_cells - 1 ? d_lattice->d_cube_id[i + 1] : d_lattice->d_cube_id[i] + 1;
    if (cube != next) d_lattice->d_cube_end[cube] = i;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::build_lattice(const Pt* __restrict__ d_X, float cube_size) {
    auto n = get_d_n();
    compute_cube_ids<<<(n + 32 - 1)/32, 32>>>(n, d_X, d_lattice, cube_size);
    thrust::fill(thrust::device, lattice.d_cube_start, lattice.d_cube_start + N_CUBES, -1);
    thrust::fill(thrust::device, lattice.d_cube_end, lattice.d_cube_end + N_CUBES, -2);
    thrust::sort_by_key(thrust::device, lattice.d_cube_id, lattice.d_cube_id + n,
        lattice.d_cell_id);
    compute_cube_start_and_end<<<(n + 32 - 1)/32, 32>>>(n, d_lattice);
}


// Integration
template<typename Pt, int N_MAX>
__global__ void calculate_lattice_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        const Lattice<N_MAX>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto Xi = d_X[d_lattice->d_cell_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = d_lattice->d_cube_id[i] + d_moore_nhood[j];
        for (auto k = d_lattice->d_cube_start[cube]; k <= d_lattice->d_cube_end[cube]; k++) {
            auto Xj = d_X[d_lattice->d_cell_id[k]];
            F += pairwise_interaction(Xi, Xj,
                d_lattice->d_cell_id[i], d_lattice->d_cell_id[k]);
        }
    }
    d_dX[d_lattice->d_cell_id[i]] = F;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::step(float delta_t, GenericForces<Pt> genforce) {
    assert(LATTICE_SIZE % 2 == 0);  // Needed?
    auto n = get_d_n();

    // 1st step
    build_lattice(d_X);
    calculate_lattice_dX<<<(n + 64 - 1)/64, 64>>>(n, d_X, d_dX, d_lattice);
    genforce(d_X, d_dX);
    euler_step<<<(n + 64 - 1)/64, 64>>>(n, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    build_lattice(d_X1);
    calculate_lattice_dX<<<(n + 64 - 1)/64, 64>>>(n, d_X1, d_dX1, d_lattice);
    genforce(d_X1, d_dX1);
    heun_step<<<(n + 64 - 1)/64, 64>>>(n, delta_t, d_X, d_dX, d_dX1);
}
