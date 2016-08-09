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


// Solution<Pt, N_MAX, Solver> combines a method Solver with a point type Pt.
// It specfies how solutions can be accessed and new steps calculated. All the action
// is happening in the Solver classes.
template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution: public Solver<Pt, N_MAX> {
public:
    Pt *h_X = Solver<Pt, N_MAX>::h_X;   // Current solution on host
    Pt *d_X = Solver<Pt, N_MAX>::d_X;   // Current solution on device (GPU)
    int *d_n = Solver<Pt, N_MAX>::d_n;  // Number of bolls
    void set_n(int n) {
        assert(n <= N_MAX);
        *Solver<Pt, N_MAX>::h_n = n;
        cudaMemcpy(Solver<Pt, N_MAX>::d_n, Solver<Pt, N_MAX>::h_n, sizeof(int),
            cudaMemcpyHostToDevice);
    }
    int get_n() {
        return *Solver<Pt, N_MAX>::h_n;
    }
    void memcpyHostToDevice() {
        cudaMemcpy(d_X, h_X, N_MAX*sizeof(Pt), cudaMemcpyHostToDevice);
        cudaMemcpy(Solver<Pt, N_MAX>::d_n, Solver<Pt, N_MAX>::h_n, sizeof(int),
            cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_X, d_X, N_MAX*sizeof(Pt), cudaMemcpyDeviceToHost);
        cudaMemcpy(Solver<Pt, N_MAX>::h_n, Solver<Pt, N_MAX>::d_n, sizeof(int),
            cudaMemcpyDeviceToHost);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint) {
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, none<Pt>);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce) {
        return Solver<Pt, N_MAX>::step(delta_t, d_pwint, genforce);
    }
};


// Integration templates
#ifdef __APPLE__
#define D_ASSERT(predicate) if (!(predicate)) printf("(%s:%d) Device assertion failed!\n", \
    __FILE__, __LINE__)
#else
#define D_ASSERT(predicate) assert(predicate)
#endif

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


// Parallelization with interactions among all pairs, after
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, int N_MAX>class N2nSolver {
protected:
    Pt *h_X = (Pt*)malloc(N_MAX*sizeof(Pt));
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    int *h_n = (int*)malloc(sizeof(int));
    int *d_n;
    N2nSolver() {
        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));

        *h_n = N_MAX;
        cudaMalloc(&d_n, sizeof(int));
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce);
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
        GenericForces<Pt> genforce) {
    // 1st step
    calculate_n2n_dX<<<(*h_n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(*h_n,
        d_X, d_dX, d_pwint);  // ceil int div.
    genforce(d_X, d_dX);
    euler_step<<<(*h_n + 32 - 1)/32, 32>>>(*h_n, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    calculate_n2n_dX<<<(*h_n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(*h_n,
        d_X1, d_dX1, d_pwint);
    genforce(d_X1, d_dX1);
    heun_step<<<(*h_n + 32 - 1)/32, 32>>>(*h_n, delta_t, d_X, d_dX, d_dX1);
}


// Sorting based lattice with limited interaction, after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
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

template<typename Pt, int N_MAX>class LatticeSolver {
public:
    Lattice<N_MAX> lattice;
    Lattice<N_MAX> *d_lattice;
    void build_lattice(float cube_size) {
        build_lattice(d_X, cube_size);
    };
protected:
    Pt *h_X = (Pt*)malloc(N_MAX*sizeof(Pt));
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    int *h_n = (int*)malloc(sizeof(int));
    int *d_n;
    LatticeSolver() {
        cudaMalloc(&d_X, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX, N_MAX*sizeof(Pt));
        cudaMalloc(&d_X1, N_MAX*sizeof(Pt));
        cudaMalloc(&d_dX1, N_MAX*sizeof(Pt));

        cudaMalloc(&d_lattice, sizeof(Lattice<N_MAX>));
        cudaMemcpy(d_lattice, &lattice, sizeof(Lattice<N_MAX>), cudaMemcpyHostToDevice);

        *h_n = N_MAX;
        cudaMalloc(&d_n, sizeof(int));
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void step(float delta_t, d_PairwiseInteraction<Pt> d_pwint, GenericForces<Pt> genforce);
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
    compute_cube_ids<<<(*h_n + 32 - 1)/32, 32>>>(*h_n, d_X, d_lattice, cube_size);
    thrust::fill(thrust::device, lattice.d_cube_start, lattice.d_cube_start + N_CUBES, -1);
    thrust::fill(thrust::device, lattice.d_cube_end, lattice.d_cube_end + N_CUBES, -2);
    thrust::sort_by_key(thrust::device, lattice.d_cube_id, lattice.d_cube_id + *h_n,
        lattice.d_cell_id);
    compute_cube_start_and_end<<<(*h_n + 32 - 1)/32, 32>>>(*h_n, d_lattice);
}


// Integration
template<typename Pt, int N_MAX>
__global__ void calculate_lattice_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        const Lattice<N_MAX>* __restrict__ d_lattice, d_PairwiseInteraction<Pt> d_pwint) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    int interacting_cubes[27];
    interacting_cubes[0] = d_lattice->d_cube_id[i] - 1;
    interacting_cubes[1] = d_lattice->d_cube_id[i];
    interacting_cubes[2] = d_lattice->d_cube_id[i] + 1;
    for (auto j = 0; j < 3; j++) {
        interacting_cubes[j + 3] = interacting_cubes[j % 3] - LATTICE_SIZE;
        interacting_cubes[j + 6] = interacting_cubes[j % 3] + LATTICE_SIZE;
    }
    for (auto j = 0; j < 9; j++) {
        interacting_cubes[j +  9] = interacting_cubes[j % 9] - LATTICE_SIZE*LATTICE_SIZE;
        interacting_cubes[j + 18] = interacting_cubes[j % 9] + LATTICE_SIZE*LATTICE_SIZE;
    }

    auto Xi = d_X[d_lattice->d_cell_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = interacting_cubes[j];
        for (auto k = d_lattice->d_cube_start[cube]; k <= d_lattice->d_cube_end[cube]; k++) {
            auto Xj = d_X[d_lattice->d_cell_id[k]];
            F += d_pwint(Xi, Xj, d_lattice->d_cell_id[i], d_lattice->d_cell_id[k]);
        }
    }
    d_dX[d_lattice->d_cell_id[i]] = F;
}

template<typename Pt, int N_MAX>
void LatticeSolver<Pt, N_MAX>::step(float delta_t, d_PairwiseInteraction<Pt> d_pwint,
        GenericForces<Pt> genforce) {
    assert(LATTICE_SIZE % 2 == 0);  // Needed?

    // 1st step
    build_lattice(d_X);
    calculate_lattice_dX<<<(*h_n + 64 - 1)/64, 64>>>(*h_n, d_X, d_dX, d_lattice, d_pwint);
    genforce(d_X, d_dX);
    euler_step<<<(*h_n + 64 - 1)/64, 64>>>(*h_n, delta_t, d_X, d_X1, d_dX);

    // 2nd step
    build_lattice(d_X1);
    calculate_lattice_dX<<<(*h_n + 64 - 1)/64, 64>>>(*h_n, d_X1, d_dX1, d_lattice, d_pwint);
    genforce(d_X1, d_dX1);
    heun_step<<<(*h_n + 64 - 1)/64, 64>>>(*h_n, delta_t, d_X, d_dX, d_dX1);
}
