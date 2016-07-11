// Solvers for N-body problem
#include <assert.h>
#include <functional>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Pt>
using PairwiseInteraction = Pt (*)(Pt Xi, Pt Xj, int i, int j);
// using PairwiseInteraction = nvstd::function<Pt (Pt Xi, Pt Xj, int i, int j)>;

template<typename Pt>
using GenericForces = std::function<void (const Pt* __restrict__ X, Pt* dX)>;

template<typename Pt>
void none(const Pt* __restrict__ X, Pt* dX) {}

// Solution<Pt, N_MAX, Solver> X; combines a method Solver with a point type Pt.
// The current solution can be accessed like Pt X[N_MAX] and the subsequent
// solution calculated with X.step(delta_t, d_pwint, genforce = none, n_cells = N_MAX).
template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution: public Solver<Pt, N_MAX> {
 public:
    __device__ __host__
    Pt& operator[](int idx) { return Solver<Pt, N_MAX>::X[idx]; }
    __device__ __host__
    const Pt& operator[](int idx) const { return Solver<Pt, N_MAX>::X[idx]; }
    void step(float delta_t, PairwiseInteraction<Pt> d_pwint, int n_cells = N_MAX) {
        assert(n_cells <= N_MAX);
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, none<Pt>, n_cells);
    }
    void step(float delta_t, PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
            int n_cells = N_MAX) {
        assert(n_cells <= N_MAX);
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, genforce, n_cells);
    }
};


// Integration templates
template<typename Pt> __global__ void euler_step(int n_cells, float delta_t,
        const Pt* __restrict__ X0, Pt* X, const Pt* __restrict__ dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    assert(dX[i].x == dX[i].x);  // For NaN f != f
    X[i] = X0[i] + dX[i]*delta_t;
}

template<typename Pt> __global__ void heun_step(int n_cells, float delta_t,
        const Pt* __restrict__ X0, Pt* X, const Pt* __restrict__ dX,
        const Pt* __restrict__ dX1) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    assert(dX1[i].x == dX1[i].x);
    X[i] = X0[i] + (dX[i] + dX1[i])*0.5*delta_t;
}


// Parallelization with interactions among all pairs, after
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, int N_MAX>class N2nSolver {
 protected:
    void step(float delta_t, PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
        int n_cells);
    Pt X[N_MAX], dX[N_MAX], X1[N_MAX], dX1[N_MAX];
};

// Calculate dX one thread per cell, to TILE_SIZE other bodies at a time
template<typename Pt>
__global__ void calculate_n2n_dX(int n_cells, const Pt* __restrict__ X, Pt* dX,
        PairwiseInteraction<Pt> d_pwint) {
    auto cell_idx = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ Pt shX[TILE_SIZE];
    auto Xi = X[cell_idx];
    Pt Fi {0};

    for (auto tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        auto other_cell_idx = tile_start + threadIdx.x;
        if (other_cell_idx < n_cells) {
            shX[threadIdx.x] = X[other_cell_idx];
        }
        __syncthreads();
        for (auto i = 0; i < TILE_SIZE; i++) {
            auto other_cell_idx = tile_start + i;
            if ((cell_idx < n_cells) and (other_cell_idx < n_cells)) {
                Fi += d_pwint(Xi, shX[i], cell_idx, other_cell_idx);
            }
        }
    }

    if (cell_idx < n_cells) {
        dX[cell_idx] = Fi;
    }
}

template<typename Pt, int N_MAX>
void N2nSolver<Pt, N_MAX>::step(float delta_t, PairwiseInteraction<Pt> d_pwint,
        GenericForces<Pt> genforce, int n_cells) {
    // 1st step
    calculate_n2n_dX<<<(n_cells + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(n_cells,
        X, dX, d_pwint);  // ceil int div.
    cudaDeviceSynchronize();
    genforce(X, dX);
    euler_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    calculate_n2n_dX<<<(n_cells + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(n_cells,
        X1, dX1, d_pwint);
    cudaDeviceSynchronize();
    genforce(X1, dX1);
    heun_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}


// Sorting based lattice with limited interaction, after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto LATTICE_SIZE = 50u;
const auto N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<typename Pt, int N_MAX>class LatticeSolver {
 public:
    void build_lattice(int n_cells, float cubesize);
    int cube_id[N_MAX], cell_id[N_MAX];
    int cube_start[N_CUBES], cube_end[N_CUBES];
 protected:
    Pt X[N_MAX], dX[N_MAX], X1[N_MAX], dX1[N_MAX];
    void build_lattice(int n_cells, const Pt* __restrict__ X, float cube_size = CUBE_SIZE);
    void step(float delta_t, PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
        int n_cells);
};


// Build lattice
template<typename Pt>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ X,
        int* cube_id, int* cell_id, float cube_size) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto id = static_cast<int>(
        (floor(X[i].x/cube_size) + LATTICE_SIZE/2) +
        (floor(X[i].y/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE +
        (floor(X[i].z/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
    assert(id >= 0);
    assert(id <= N_CUBES);
    cube_id[i] = id;
    cell_id[i] = i;
}

__global__ void compute_cube_start_and_end(int n_cells, const int* __restrict__ cube_id,
        int* cube_start, int* cube_end) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto cube = cube_id[i];
    auto prev = i > 0 ? cube_id[i - 1] : -1;
    if (cube != prev) cube_start[cube] = i;
    auto next = i < n_cells ? cube_id[i + 1] : cube_id[i] + 1;
    if (cube != next) cube_end[cube] = i;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::build_lattice(int n_cells, const Pt* __restrict__ X,
        float cube_size) {
    assert(n_cells <= N_MAX);
    compute_cube_ids<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, X, cube_id, cell_id, cube_size);
    cudaDeviceSynchronize();
    thrust::fill(thrust::device, cube_start, cube_start + N_CUBES, -1);
    thrust::fill(thrust::device, cube_end, cube_end + N_CUBES, -2);
    thrust::sort_by_key(thrust::device, cube_id, cube_id + n_cells, cell_id);
    compute_cube_start_and_end<<<(n_cells + 32 - 1)/32, 32>>>(n_cells,
        cube_id, cube_start, cube_end);
    cudaDeviceSynchronize();
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::build_lattice(int n_cells, float cube_size) {
    build_lattice(n_cells, X, cube_size);
}


// Integration
template<typename Pt>
__global__ void calculate_lattice_dX(int n_cells, const Pt* __restrict__ X, Pt* dX,
        const int* __restrict__ cell_id, const int* __restrict__ cube_id,
        const int* __restrict__ cube_start, const int* __restrict__ cube_end,
        PairwiseInteraction<Pt> d_pwint) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    int interacting_cubes[27];
    interacting_cubes[0] = cube_id[i] - 1;
    interacting_cubes[1] = cube_id[i];
    interacting_cubes[2] = cube_id[i] + 1;
    for (auto j = 0; j < 3; j++) {
        interacting_cubes[j + 3] = interacting_cubes[j % 3] - LATTICE_SIZE;
        interacting_cubes[j + 6] = interacting_cubes[j % 3] + LATTICE_SIZE;
    }
    for (auto j = 0; j < 9; j++) {
        interacting_cubes[j +  9] = interacting_cubes[j % 9] - LATTICE_SIZE*LATTICE_SIZE;
        interacting_cubes[j + 18] = interacting_cubes[j % 9] + LATTICE_SIZE*LATTICE_SIZE;
    }

    auto Xi = X[cell_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = interacting_cubes[j];
        for (auto k = cube_start[cube]; k <= cube_end[cube]; k++) {
            auto Xj = X[cell_id[k]];
            F += d_pwint(Xi, Xj, cell_id[i], cell_id[k]);
        }
    }
    dX[cell_id[i]] = F;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::step(float delta_t, PairwiseInteraction<Pt> d_pwint,
        GenericForces<Pt> genforce, int n_cells) {
    assert(LATTICE_SIZE % 2 == 0);  // Needed?

    // 1st step
    build_lattice(n_cells, X);
    calculate_lattice_dX<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, X, dX,
        cell_id, cube_id, cube_start, cube_end, d_pwint);
    cudaDeviceSynchronize();
    genforce(X, dX);
    euler_step<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, delta_t, X, X1, dX);
    cudaDeviceSynchronize();

    // 2nd step
    build_lattice(n_cells, X1);
    calculate_lattice_dX<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, X1, dX1,
        cell_id, cube_id, cube_start, cube_end, d_pwint);
    cudaDeviceSynchronize();
    genforce(X1, dX1);
    heun_step<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, delta_t, X, X, dX, dX1);
    cudaDeviceSynchronize();
}
