// Solvers for N-body problems
#pragma once

#include <assert.h>
#include <functional>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cudebug.cuh"


// Interactions must be specified between two points of type Pt (e.g. float3,
// see dtypes.cuh). Pt contains the variables to be integrated, e.g. position
// or concentrations.
template<typename Pt>
using Pairwise_interaction = Pt (Pt Xi, Pt Xj, int i, int j);

// In addition a generic force can be passed optionally:
template<typename Pt>
using Generic_forces = std::function<void (const Pt* __restrict__ d_X, Pt* d_dX)>;

template<typename Pt>
void no_gen_forces(const Pt* __restrict__ d_X, Pt* d_dX) {}


// Solution<Pt, n_max, Solver> combines a method, Solver, with a point type, Pt.
// It stores the variables on the host and specifies how the variables on the
// device can be accessed and how new steps are computed. However, all the GPU
// action is happening in the Solver classes.
template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution: public Solver<Pt, n_max> {
public:
    Pt *h_X = (Pt*)malloc(n_max*sizeof(Pt));  // Current variables on host
    Pt *d_X = Solver<Pt, n_max>::d_X;         // Variables on device (GPU)
    int *h_n = (int*)malloc(sizeof(int));     // Number of bolls
    int *d_n = Solver<Pt, n_max>::d_n;
    Solution(int n_0 = n_max) {
        *h_n = n_0;
    }
    void copy_to_device() {
        assert(*h_n <= n_max);
        cudaMemcpy(d_X, h_X, n_max*sizeof(Pt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void copy_to_host() {
        cudaMemcpy(h_X, d_X, n_max*sizeof(Pt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= n_max);
    }
    int get_d_n() {
        return Solver<Pt, n_max>::get_d_n();
    }
    template<Pairwise_interaction<Pt> pw_int>
    void take_step(float dt, Generic_forces<Pt> gen_forces = no_gen_forces<Pt>) {
        return Solver<Pt, n_max>::template take_step<pw_int>(dt, gen_forces);
    }
};


// Integration templates
template<typename Pt> __global__ void euler_step(int n_cells, float dt,
        const Pt* __restrict__ d_X0, Pt* d_X, Pt* d_dX,
        const float3* __restrict__ d_sum_v, const int* __restrict__ d_nNBs,
        float3* d_old_v) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    D_ASSERT(d_dX[i].x == d_dX[i].x);  // For NaN f != f
    d_dX[i].x += d_sum_v[i].x/d_nNBs[i];
    d_dX[i].y += d_sum_v[i].y/d_nNBs[i];
    d_dX[i].z += d_sum_v[i].z/d_nNBs[i];
    d_X[i] = d_X0[i] + d_dX[i]*dt;

    // d_old_v[i].x = d_dX[i].x/1.0;
    // d_old_v[i].y = d_dX[i].y/1.0;
    // d_old_v[i].z = d_dX[i].z/1.0;
}

template<typename Pt> __global__ void heun_step(int n_cells, float dt,
        Pt* d_X, const Pt* __restrict__ d_dX, Pt* d_dX1,
        const float3* __restrict__ d_sum_v, const int* __restrict__ d_nNBs,
        float3* d_old_v) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    D_ASSERT(d_dX1[i].x == d_dX1[i].x);
    d_dX1[i].x += d_sum_v[i].x/d_nNBs[i];
    d_dX1[i].y += d_sum_v[i].y/d_nNBs[i];
    d_dX1[i].z += d_sum_v[i].z/d_nNBs[i];
    d_X[i] += (d_dX[i] + d_dX1[i])*0.5*dt;

    d_X[i].v = norm3df(d_sum_v[i].x/d_nNBs[i], d_sum_v[i].y/d_nNBs[i], d_sum_v[i].z/d_nNBs[i]);

    d_old_v[i].x = (d_dX[i].x + d_dX1[i].x)*0.5/1.0;
    d_old_v[i].y = (d_dX[i].y + d_dX1[i].y)*0.5/1.0;
    d_old_v[i].z = (d_dX[i].z + d_dX1[i].z)*0.5/1.0;
}


// Solver implementation with interactions among all pairs, after
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, int n_max>class N2n_solver {
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    float3 *d_old_v, *d_sum_v;
    int *d_n, *d_nNBs;
    N2n_solver() {
        cudaMalloc(&d_X, n_max*sizeof(Pt));
        cudaMalloc(&d_dX, n_max*sizeof(Pt));
        cudaMalloc(&d_X1, n_max*sizeof(Pt));
        cudaMalloc(&d_dX1, n_max*sizeof(Pt));

        cudaMalloc(&d_old_v, n_max*sizeof(float3));
        thrust::fill(thrust::device, d_old_v, d_old_v + n_max, float3 {0});
        cudaMalloc(&d_sum_v, n_max*sizeof(float3)); CHECK_CUDA;

        cudaMalloc(&d_n, sizeof(int)); CHECK_CUDA;
        cudaMalloc(&d_nNBs, n_max*sizeof(int));
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost); CHECK_CUDA;
        assert(n <= n_max);
        return n;
    }
    template<Pairwise_interaction<Pt> pw_int>
    void take_step(float dt, Generic_forces<Pt> gen_forces);
};

__global__ void keep_one_fixed(int n_cells, float3* d_old_v) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    d_old_v[i] -= d_old_v[0];
}

// Calculate d_dX one thread per point, to TILE_SIZE other points at a time
template<typename Pt, Pairwise_interaction<Pt> pw_int>
__global__ void compute_n2n_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        const float3* __restrict__ d_old_v, float3* d_sum_v, int* d_nNBs) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ Pt shX[TILE_SIZE];

    Pt Fi {0};
    for (auto tile_start = 0; tile_start < n_cells; tile_start += TILE_SIZE) {
        auto j = tile_start + threadIdx.x;
        if (j < n_cells) {
            shX[threadIdx.x] = d_X[j];
        }
        __syncthreads();

        for (auto k = 0; k < TILE_SIZE; k++) {
            auto j = tile_start + k;
            if ((i < n_cells) and (j < n_cells)) {
                auto r = d_X[i] - shX[k];
                auto dist = norm3df(r.x, r.y, r.z);
                if (dist < 1) {
                    d_nNBs[i] += 1;
                    d_sum_v[i] += d_old_v[j];
                    Fi += pw_int(d_X[i], shX[k], i, j);
                }
            }
        }
    }

    if (i < n_cells) {
        d_dX[i] = Fi;
    }
}

template<typename Pt, int n_max>
template<Pairwise_interaction<Pt> pw_int>
void N2n_solver<Pt, n_max>::take_step(float dt, Generic_forces<Pt> gen_forces) {
    auto n = get_d_n();

    // 1st step
    thrust::fill(thrust::device, d_nNBs, d_nNBs + n, 0);
    thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
    keep_one_fixed<<<(n + 64 - 1)/64, 64>>>(n, d_old_v);
    compute_n2n_dX<Pt, pw_int><<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(  // ceil int div.
        n, d_X, d_dX, d_old_v, d_sum_v, d_nNBs);
    gen_forces(d_X, d_dX);
    euler_step<<<(n + 32 - 1)/32, 32>>>(n, dt, d_X, d_X1, d_dX, d_sum_v, d_nNBs, d_old_v);

    // 2nd step
    thrust::fill(thrust::device, d_nNBs, d_nNBs + n, 0);
    thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
    keep_one_fixed<<<(n + 64 - 1)/64, 64>>>(n, d_old_v);
    compute_n2n_dX<Pt, pw_int><<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(
        n, d_X1, d_dX1, d_old_v, d_sum_v, d_nNBs);
    gen_forces(d_X1, d_dX1);
    heun_step<<<(n + 32 - 1)/32, 32>>>(n, dt, d_X, d_dX, d_dX1, d_sum_v, d_nNBs, d_old_v);
}


// Solver implementation with sorting based lattice for limited pairwise_interaction,
// after http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto LATTICE_SIZE = 50;
const auto N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<int n_max>struct Lattice {
public:
    int *d_cube_id, *d_cell_id, *d_cube_start, *d_cube_end;
    Lattice() {
        cudaMalloc(&d_cube_id, n_max*sizeof(int));
        cudaMalloc(&d_cell_id, n_max*sizeof(int));
        cudaMalloc(&d_cube_start, N_CUBES*sizeof(int));
        cudaMalloc(&d_cube_end, N_CUBES*sizeof(int));
    }
};

__constant__ int d_moore_nhood[27];  // Yes, this is a waste if no Lattice_solver is used

template<typename Pt, int n_max>class Lattice_solver {
public:
    Lattice<n_max> lattice;
    Lattice<n_max> *d_lattice;
    void build_lattice(float cube_size) {
        build_lattice(d_X, cube_size);
    };
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    float3 *d_old_v, *d_sum_v;
    int *d_n, *d_nNBs;
    Lattice_solver() {
        cudaMalloc(&d_lattice, sizeof(Lattice<n_max>));
        cudaMemcpy(d_lattice, &lattice, sizeof(Lattice<n_max>), cudaMemcpyHostToDevice);

        cudaMalloc(&d_X, n_max*sizeof(Pt));
        cudaMalloc(&d_dX, n_max*sizeof(Pt));
        cudaMalloc(&d_X1, n_max*sizeof(Pt));
        cudaMalloc(&d_dX1, n_max*sizeof(Pt));

        cudaMalloc(&d_old_v, n_max*sizeof(float3));
        thrust::fill(thrust::device, d_old_v, d_old_v + n_max, float3 {0});
        cudaMalloc(&d_sum_v, n_max*sizeof(float3));

        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_nNBs, n_max*sizeof(int));

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
        assert(n <= n_max);
        return n;
    }
    template<Pairwise_interaction<Pt> pw_int>
    void take_step(float dt, Generic_forces<Pt> gen_forces);
    void build_lattice(const Pt* __restrict__ d_X, float cube_size = CUBE_SIZE);
};


// Build lattice
template<typename Pt, int n_max>
__global__ void compute_cube_ids(int n_cells, const Pt* __restrict__ d_X,
        Lattice<n_max>* d_lattice, float cube_size) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x/cube_size) + LATTICE_SIZE/2) +
        (floor(d_X[i].y/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE +
        (floor(d_X[i].z/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
    D_ASSERT(id >= 0);
    D_ASSERT(id < N_CUBES);
    d_lattice->d_cube_id[i] = id;
    d_lattice->d_cell_id[i] = i;
}

template<int n_max>
__global__ void compute_cube_start_and_end(int n_cells, Lattice<n_max>* d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto cube = d_lattice->d_cube_id[i];
    auto prev = i > 0 ? d_lattice->d_cube_id[i - 1] : -1;
    if (cube != prev) d_lattice->d_cube_start[cube] = i;
    auto next = i < n_cells - 1 ? d_lattice->d_cube_id[i + 1] : d_lattice->d_cube_id[i] + 1;
    if (cube != next) d_lattice->d_cube_end[cube] = i;
}

template<typename Pt, int n_max>
void Lattice_solver<Pt, n_max>::build_lattice(const Pt* __restrict__ d_X, float cube_size) {
    auto n = get_d_n();
    compute_cube_ids<<<(n + 32 - 1)/32, 32>>>(n, d_X, d_lattice, cube_size);
    thrust::fill(thrust::device, lattice.d_cube_start, lattice.d_cube_start + N_CUBES, -1);
    thrust::fill(thrust::device, lattice.d_cube_end, lattice.d_cube_end + N_CUBES, -2);
    thrust::sort_by_key(thrust::device, lattice.d_cube_id, lattice.d_cube_id + n,
        lattice.d_cell_id);
    compute_cube_start_and_end<<<(n + 32 - 1)/32, 32>>>(n, d_lattice);
}


// Integration
template<typename Pt, int n_max, Pairwise_interaction<Pt> pw_int>
__global__ void compute_lattice_dX(int n_cells, const Pt* __restrict__ d_X, Pt* d_dX,
        const float3* __restrict__ d_old_v, float3* d_sum_v, int* d_nNBs,
        const Lattice<n_max>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    auto Xi = d_X[d_lattice->d_cell_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = d_lattice->d_cube_id[i] + d_moore_nhood[j];
        for (auto k = d_lattice->d_cube_start[cube]; k <= d_lattice->d_cube_end[cube]; k++) {
            auto Xj = d_X[d_lattice->d_cell_id[k]];
            auto r = Xi - Xj;
            auto dist = norm3df(r.x, r.y, r.z);
            if (dist < CUBE_SIZE) {
                d_nNBs[d_lattice->d_cell_id[i]] += 1;
                d_sum_v[d_lattice->d_cell_id[i]] += d_old_v[d_lattice->d_cell_id[k]];
                F += pw_int(Xi, Xj, d_lattice->d_cell_id[i], d_lattice->d_cell_id[k]);
            }
        }
    }
    d_dX[d_lattice->d_cell_id[i]] = F;
}

template<typename Pt, int n_max>
template<Pairwise_interaction<Pt> pw_int>
void Lattice_solver<Pt, n_max>::take_step(float dt, Generic_forces<Pt> gen_forces) {
    assert(LATTICE_SIZE % 2 == 0);  // Needed?
    auto n = get_d_n();

    // 1st step
    build_lattice(d_X);
    thrust::fill(thrust::device, d_nNBs, d_nNBs + n, 0);
    thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
    keep_one_fixed<<<(n + 64 - 1)/64, 64>>>(n, d_old_v);
    compute_lattice_dX<Pt, n_max, pw_int><<<(n + 64 - 1)/64, 64>>>(n, d_X, d_dX,
        d_old_v, d_sum_v, d_nNBs, d_lattice);
    gen_forces(d_X, d_dX);
    euler_step<<<(n + 64 - 1)/64, 64>>>(n, dt, d_X, d_X1, d_dX, d_sum_v, d_nNBs, d_old_v);

    // 2nd step
    build_lattice(d_X1);
    thrust::fill(thrust::device, d_nNBs, d_nNBs + n, 0);
    thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
    keep_one_fixed<<<(n + 64 - 1)/64, 64>>>(n, d_old_v);
    compute_lattice_dX<Pt, n_max, pw_int><<<(n + 64 - 1)/64, 64>>>(n, d_X1, d_dX1,
        d_old_v, d_sum_v, d_nNBs, d_lattice);
    gen_forces(d_X1, d_dX1);
    heun_step<<<(n + 64 - 1)/64, 64>>>(n, dt, d_X, d_dX, d_dX1, d_sum_v, d_nNBs, d_old_v);
}
