// Solvers for N-body problem
#include <assert.h>
#include <functional>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Pt>
using d_PairwiseInteraction = Pt (*)(Pt Xi, Pt Xj, int i, int j);

template<typename Pt>
using GenericForces = std::function<void (const Pt* __restrict__ d_X, Pt* d_dX)>;

template<typename Pt>
void none(const Pt* __restrict__ d_X, Pt* d_dX) {}

// Wrapper to copy device object to host, from pisto
template<typename T> T get_device_object(const T& on_device, cudaStream_t stream = 0){
    T on_host;
    cudaMemcpyFromSymbolAsync((void*)&on_host, (const void*)&on_device, sizeof(T), 0,
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return on_host;
}


// Solution<Pt, N_MAX, Solver> combines a method Solver with a point type Pt. It provides
// pointers h_X and d_X to the current solution, methods to copy the solution from host to
// device and vice versa, and a method step() to calculate the next solution. All the
// action is happening in the solver class.
template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution: public Solver<Pt, N_MAX> {
public:
    Pt *h_X = Solver<Pt, N_MAX>::h_X;
    Pt *d_X = Solver<Pt, N_MAX>::d_X;
    void memcpyHostToDevice() {
        cudaMemcpy(d_X, h_X, N_MAX*sizeof(Pt), cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_X, d_X, N_MAX*sizeof(Pt), cudaMemcpyDeviceToHost);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, int n_cells = N_MAX) {
        assert(n_cells <= N_MAX);
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, none<Pt>, n_cells);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
            int n_cells = N_MAX) {
        assert(n_cells <= N_MAX);
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, genforce, n_cells);
    }
};


// Integration templates
template<typename Pt> __global__ void euler_step(int n_cells, float delta_t,
        const Pt* __restrict__ d_X0, Pt* d_X, const Pt* __restrict__ d_dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    assert(d_dX[i].x == d_dX[i].x);  // For NaN f != f
    d_X[i] = d_X0[i] + d_dX[i]*delta_t;
}

template<typename Pt> __global__ void heun_step(int n_cells, float delta_t,
        Pt* d_X, const Pt* __restrict__ d_dX, const Pt* __restrict__ d_dX1) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    assert(d_dX1[i].x == d_dX1[i].x);
    d_X[i] += (d_dX[i] + d_dX1[i])*0.5*delta_t;
}


// Parallelization with interactions among all pairs, after
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, int N_MAX>class N2nSolver {
protected:
    Pt *h_X = (Pt*)malloc(N_MAX*sizeof(Pt));
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    int h_n;
    int *d_n;
    N2nSolver() {
        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
        int n_cells);
};

// Calculate d_dX one thread per cell, to TILE_SIZE other bodies at a time
template<typename Pt>
__global__ void calculate_n2n_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        d_PairwiseInteraction<Pt> d_pwint) {
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
                Fi += d_pwint(Xi, shX[i], d_cell_idx, other_d_cell_idx);
            }
        }
    }

    if (d_cell_idx < n_cells) {
        d_dX[d_cell_idx] = Fi;
    }
}

template<typename Pt, int N_MAX>
void N2nSolver<Pt, N_MAX>::step(float delta_t, d_PairwiseInteraction<Pt> d_pwint,
        GenericForces<Pt> genforce, int n_cells) {
    // 1st step
    calculate_n2n_dX<<<(n_cells + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(n_cells,
        d_X, d_dX, d_pwint);  // ceil int div.
    genforce(d_X, d_dX);
    euler_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    calculate_n2n_dX<<<(n_cells + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(n_cells,
        d_X1, d_dX1, d_pwint);
    genforce(d_X1, d_dX1);
    heun_step<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, delta_t, d_X, d_dX, d_dX1);
}


// Sorting based lattice with limited interaction, after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto LATTICE_SIZE = 50u;
const auto N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<typename Pt, int N_MAX>class LatticeSolver {
public:
    int *d_cube_id, *d_cell_id, *d_cube_start, *d_cube_end;
    void build_lattice(int n_cells, float cube_size) {
        build_lattice(n_cells, d_X, cube_size);
    };
protected:
    Pt *h_X = (Pt*)malloc(N_MAX*sizeof(Pt));
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    LatticeSolver() {
        cudaMalloc(&d_cube_id, N_MAX*sizeof(int));
        cudaMalloc(&d_cell_id, N_MAX*sizeof(int));
        cudaMalloc(&d_cube_start, N_CUBES*sizeof(int));
        cudaMalloc(&d_cube_end, N_CUBES*sizeof(int));

        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));
    }
    void build_lattice(int n_cells, const Pt* __restrict__ d_X, float cube_size = CUBE_SIZE);
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce,
        int n_cells);
};


// Build lattice
template<typename Pt>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ d_X,
        int* d_cube_id, int* d_cell_id, float cube_size) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x/cube_size) + LATTICE_SIZE/2) +
        (floor(d_X[i].y/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE +
        (floor(d_X[i].z/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
    assert(id >= 0);
    assert(id <= N_CUBES);
    d_cube_id[i] = id;
    d_cell_id[i] = i;
}

__global__ void compute_cube_start_and_end(int n_cells, const int* __restrict__ d_cube_id,
        int* d_cube_start, int* d_cube_end) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto cube = d_cube_id[i];
    auto prev = i > 0 ? d_cube_id[i - 1] : -1;
    if (cube != prev) d_cube_start[cube] = i;
    auto next = i < n_cells ? d_cube_id[i + 1] : d_cube_id[i] + 1;
    if (cube != next) d_cube_end[cube] = i;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::build_lattice(int n_cells, const Pt* __restrict__ d_X,
        float cube_size) {
    assert(n_cells <= N_MAX);
    compute_cube_ids<<<(n_cells + 32 - 1)/32, 32>>>(n_cells, d_X, d_cube_id, d_cell_id,
        cube_size);
    thrust::fill(thrust::device, d_cube_start, d_cube_start + N_CUBES, -1);
    thrust::fill(thrust::device, d_cube_end, d_cube_end + N_CUBES, -2);
    thrust::sort_by_key(thrust::device, d_cube_id, d_cube_id + n_cells, d_cell_id);
    compute_cube_start_and_end<<<(n_cells + 32 - 1)/32, 32>>>(n_cells,
        d_cube_id, d_cube_start, d_cube_end);
}


// Integration
template<typename Pt>
__global__ void calculate_lattice_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        const int* __restrict__ d_cell_id, const int* __restrict__ d_cube_id,
        const int* __restrict__ d_cube_start, const int* __restrict__ d_cube_end,
        d_PairwiseInteraction<Pt> d_pwint) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    int interacting_cubes[27];
    interacting_cubes[0] = d_cube_id[i] - 1;
    interacting_cubes[1] = d_cube_id[i];
    interacting_cubes[2] = d_cube_id[i] + 1;
    for (auto j = 0; j < 3; j++) {
        interacting_cubes[j + 3] = interacting_cubes[j % 3] - LATTICE_SIZE;
        interacting_cubes[j + 6] = interacting_cubes[j % 3] + LATTICE_SIZE;
    }
    for (auto j = 0; j < 9; j++) {
        interacting_cubes[j +  9] = interacting_cubes[j % 9] - LATTICE_SIZE*LATTICE_SIZE;
        interacting_cubes[j + 18] = interacting_cubes[j % 9] + LATTICE_SIZE*LATTICE_SIZE;
    }

    auto Xi = d_X[d_cell_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = interacting_cubes[j];
        for (auto k = d_cube_start[cube]; k <= d_cube_end[cube]; k++) {
            auto Xj = d_X[d_cell_id[k]];
            F += d_pwint(Xi, Xj, d_cell_id[i], d_cell_id[k]);
        }
    }
    d_dX[d_cell_id[i]] = F;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::step(float delta_t, d_PairwiseInteraction<Pt> d_pwint,
        GenericForces<Pt> genforce, int n_cells) {
    assert(LATTICE_SIZE % 2 == 0);  // Needed?

    // 1st step
    build_lattice(n_cells, d_X);
    calculate_lattice_dX<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, d_X, d_dX,
        d_cell_id, d_cube_id, d_cube_start, d_cube_end, d_pwint);
    genforce(d_X, d_dX);
    euler_step<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    build_lattice(n_cells, d_X1);
    calculate_lattice_dX<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, d_X1, d_dX1,
        d_cell_id, d_cube_id, d_cube_start, d_cube_end, d_pwint);
    genforce(d_X1, d_dX1);
    heun_step<<<(n_cells + 64 - 1)/64, 64>>>(n_cells, delta_t, d_X, d_dX, d_dX1);
}
