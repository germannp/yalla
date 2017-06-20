// Solvers for N-body problems
#pragma once

#include <assert.h>
#include <functional>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "cudebug.cuh"


// Interactions must be specified between two points Xi and Xj, with  r = Xi - Xj.
// The type Pt (e.g. float3, see dtypes.cuh) contains the variables to be integrated,
// e.g. position or concentrations.
template<typename Pt>
using Pairwise_interaction = Pt (Pt Xi, Pt r, float dist, int i, int j);

// Similarly, a pairwise friction coefficient can be specified, see
// http://dx.doi.org/10.1007/s10237-014-0613-5. By default bolls closer
// than 1 exert friction on each other.
template<typename Pt>
using Pairwise_friction = float (Pt Xi, Pt r, float dist, int i, int j);

template<typename Pt>
__device__ float constant_friction(Pt Xi, Pt r, float dist, int i, int j) {
    if (i == j) return 0;
    if (dist < 1) return 1;
    return 0;
}

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
    int *h_n = (int*)malloc(sizeof(int));     // Number of points
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
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction = constant_friction<Pt>>
    void take_step(float dt, Generic_forces<Pt> gen_forces = no_gen_forces<Pt>) {
        return Solver<Pt, n_max>::template take_step<pw_int, pw_friction>(dt, gen_forces);
    }
};


// 2nd order solver for the equation v = F + <v(t - dt)> for x, y, and z, where
// <v> is the mean velocity of the neighbours weighted by the friction coefficients.
// The center of mass is kept fixed. Solves dw/dt = F_w for other variables in Pt.
template<typename Pt> __global__ void euler_step(const int n, const float dt,
        const Pt* __restrict__ d_X0, const Pt mean_dX, Pt* d_dX, Pt* d_X) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX[i].x -= mean_dX.x;
    d_dX[i].y -= mean_dX.y;
    d_dX[i].z -= mean_dX.z;

    d_X[i] = d_X0[i] + d_dX[i]*dt;
}

template<typename Pt> __global__ void heun_step(const int n, const float dt,
        const Pt* __restrict__ d_dX, const Pt mean_dX1, Pt* d_dX1, Pt* d_X, float3* d_old_v) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX1[i].x -= mean_dX1.x;
    d_dX1[i].y -= mean_dX1.y;
    d_dX1[i].z -= mean_dX1.z;

    d_X[i] += (d_dX[i] + d_dX1[i])*0.5*dt;

    d_old_v[i].x = (d_dX[i].x + d_dX1[i].x)*0.5;
    d_old_v[i].y = (d_dX[i].y + d_dX1[i].y)*0.5;
    d_old_v[i].z = (d_dX[i].z + d_dX1[i].z)*0.5;
}

template<typename Pt> __global__ void add_rhs(const int n, const float3* __restrict__ d_sum_v,
        const float* __restrict__ d_sum_friction, Pt* d_dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    D_ASSERT(d_dX[i].x == d_dX[i].x);  // For NaN f != f
    D_ASSERT(d_sum_v[i].x == d_sum_v[i].x);

    if (d_sum_friction[i] > 0) {
        d_dX[i].x += d_sum_v[i].x/d_sum_friction[i];
        d_dX[i].y += d_sum_v[i].y/d_sum_friction[i];
        d_dX[i].z += d_sum_v[i].z/d_sum_friction[i];
    }
}

template<typename Pt, int n_max, template<typename, int> class Computer>
class Order2_solver: public Computer<Pt, n_max> {
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    float3 *d_old_v, *d_sum_v;
    float *d_sum_friction;
    int *d_n;
    Order2_solver() {
        cudaMalloc(&d_X, n_max*sizeof(Pt));
        cudaMalloc(&d_dX, n_max*sizeof(Pt));
        cudaMalloc(&d_X1, n_max*sizeof(Pt));
        cudaMalloc(&d_dX1, n_max*sizeof(Pt));

        cudaMalloc(&d_old_v, n_max*sizeof(float3));
        thrust::fill(thrust::device, d_old_v, d_old_v + n_max, float3 {0});
        cudaMalloc(&d_sum_v, n_max*sizeof(float3));

        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_sum_friction, n_max*sizeof(int));
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= n_max);
        return n;
    }
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void take_step(float dt, Generic_forces<Pt> gen_forces) {
        auto n = get_d_n();

        // 1st step
        thrust::fill(thrust::device, d_sum_friction, d_sum_friction + n, 0);
        thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
        Computer<Pt, n_max>::template pwints<pw_int, pw_friction>(
            n, d_X, d_dX, d_old_v, d_sum_v, d_sum_friction);
        gen_forces(d_X, d_dX);
        add_rhs<<<(n + 32 - 1)/32, 32>>>(n, d_sum_v, d_sum_friction, d_dX);  // ceil int div.
        auto mean_dX = thrust::reduce(thrust::device, d_dX, d_dX + n, Pt {0})/n;
        euler_step<<<(n + 32 - 1)/32, 32>>>(n, dt, d_X, mean_dX, d_dX, d_X1);

        // 2nd step
        thrust::fill(thrust::device, d_sum_friction, d_sum_friction + n, 0);
        thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3 {0});
        Computer<Pt, n_max>::template pwints<pw_int, pw_friction>(
            n, d_X1, d_dX1, d_old_v, d_sum_v, d_sum_friction);
        gen_forces(d_X1, d_dX1);
        add_rhs<<<(n + 32 - 1)/32, 32>>>(n, d_sum_v, d_sum_friction, d_dX1);
        auto mean_dX1 = thrust::reduce(thrust::device, d_dX1, d_dX1 + n, Pt {0})/n;
        heun_step<<<(n + 32 - 1)/32, 32>>>(n, dt, d_dX, mean_dX1, d_dX1, d_X, d_old_v);
    }
};


// Compute pairwise interactions and frictions one thread per point, to TILE_SIZE points
// at a time, after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
__global__ void compute_tiles(int n, const Pt* __restrict__ d_X, Pt* d_dX,
        const float3* __restrict__ d_old_v, float3* d_sum_v, float* d_sum_friction) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ Pt shX[TILE_SIZE];

    Pt Fi {0};
    for (auto tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        auto j = tile_start + threadIdx.x;
        if (j < n) {
            shX[threadIdx.x] = d_X[j];
        }
        __syncthreads();

        for (auto k = 0; k < TILE_SIZE; k++) {
            auto j = tile_start + k;
            if ((i < n) and (j < n)) {
                auto r = d_X[i] - shX[k];
                auto dist = norm3df(r.x, r.y, r.z);
                Fi += pw_int(d_X[i], r, dist, i, j);
                auto friction = pw_friction(d_X[i], r, dist, i, j);
                d_sum_friction[i] += friction;
                d_sum_v[i] += friction*d_old_v[j];
            }
        }
    }

    if (i < n) {
        d_dX[i] = Fi;
    }
}

template<typename Pt, int n_max> class N2n_computer {
protected:
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(int n, Pt* d_X, Pt* d_dX, const float3* __restrict__ d_old_v,
            float3* d_sum_v, float* d_sum_friction) {
        compute_tiles<Pt, pw_int, pw_friction><<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(
            n, d_X, d_dX, d_old_v, d_sum_v, d_sum_friction);
    }
};

template<typename Pt, int n_max> using N2n_solver = Order2_solver<Pt, n_max, N2n_computer>;


// Compute pairwise interactions and frictions with sorting based lattice ONLY for
// bolls closer than CUBE_SIZE. Scales linearly in n, faster with maybe 7k bolls.
// After http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/
// projects/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto LATTICE_SIZE = 50;
const auto N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;

template<typename Pt>
__global__ void compute_cube_ids(int n, const Pt* __restrict__ d_X,
        int* d_cube_id, int* d_point_id, float cube_size) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x/cube_size) + LATTICE_SIZE/2) +
        (floor(d_X[i].y/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE +
        (floor(d_X[i].z/cube_size) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
    D_ASSERT(id >= 0);
    D_ASSERT(id < N_CUBES);
    d_cube_id[i] = id;
    d_point_id[i] = i;
}

__global__ void compute_cube_start_and_end(int n, int* d_cube_id,
        int* d_cube_start, int* d_cube_end) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto cube = d_cube_id[i];
    auto prev = i > 0 ? d_cube_id[i - 1] : -1;
    if (cube != prev) d_cube_start[cube] = i;
    auto next = i < n - 1 ? d_cube_id[i + 1] : d_cube_id[i] + 1;
    if (cube != next) d_cube_end[cube] = i;
}

template<int n_max> class Lattice {
public:
    int *d_cube_id, *d_point_id, *d_cube_start, *d_cube_end;
    Lattice() {
        cudaMalloc(&d_cube_id, n_max*sizeof(int));
        cudaMalloc(&d_point_id, n_max*sizeof(int));
        cudaMalloc(&d_cube_start, N_CUBES*sizeof(int));
        cudaMalloc(&d_cube_end, N_CUBES*sizeof(int));
    }
    template<typename Pt>
    void build(int n, const Pt* __restrict__ d_X, float cube_size = CUBE_SIZE) {
        compute_cube_ids<<<(n + 32 - 1)/32, 32>>>(n, d_X, d_cube_id, d_point_id, cube_size);
        thrust::fill(thrust::device, d_cube_start, d_cube_start + N_CUBES, -1);
        thrust::fill(thrust::device, d_cube_end, d_cube_end + N_CUBES, -2);
        thrust::sort_by_key(thrust::device, d_cube_id, d_cube_id + n, d_point_id);
        compute_cube_start_and_end<<<(n + 32 - 1)/32, 32>>>(n, d_cube_id, d_cube_start, d_cube_end);
    }
    template<typename Pt, int n_max_solution, template<typename, int> class Solver>
    void build(Solution<Pt, n_max_solution, Solver>& bolls, float cube_size = CUBE_SIZE) {
        auto n = bolls.get_d_n();
        assert(n <= n_max);
        build(n, bolls.d_X, cube_size);
    }
};


__constant__ int d_moore_nhood[27];  // This is wasted if no Lattice_computer is used

template<typename Pt, int n_max, Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
__global__ void compute_lattice_pwints(int n, const Pt* __restrict__ d_X, Pt* d_dX,
        const float3* __restrict__ d_old_v, float3* d_sum_v, float* d_sum_friction,
        const Lattice<n_max>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto Xi = d_X[d_lattice->d_point_id[i]];
    Pt F {0};
    for (auto j = 0; j < 27; j++) {
        auto cube = d_lattice->d_cube_id[i] + d_moore_nhood[j];
        for (auto k = d_lattice->d_cube_start[cube]; k <= d_lattice->d_cube_end[cube]; k++) {
            auto Xj = d_X[d_lattice->d_point_id[k]];
            auto r = Xi - Xj;
            auto dist = norm3df(r.x, r.y, r.z);
            if (dist < CUBE_SIZE) {
                F += pw_int(Xi, r, dist, d_lattice->d_point_id[i], d_lattice->d_point_id[k]);
                auto friction = pw_friction(Xi, r, dist, d_lattice->d_point_id[i], d_lattice->d_point_id[k]);
                d_sum_friction[d_lattice->d_point_id[i]] += friction;
                d_sum_v[d_lattice->d_point_id[i]] += friction*d_old_v[d_lattice->d_point_id[k]];
            }
        }
    }
    d_dX[d_lattice->d_point_id[i]] = F;
}

template<typename Pt, int n_max> class Lattice_computer {
protected:
    Lattice<n_max> lattice;
    Lattice<n_max> *d_lattice;
    Lattice_computer() {
        cudaMalloc(&d_lattice, sizeof(Lattice<n_max>));
        cudaMemcpy(d_lattice, &lattice, sizeof(Lattice<n_max>), cudaMemcpyHostToDevice);

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
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(int n, Pt* d_X, Pt* d_dX, const float3* __restrict__ d_old_v,
            float3* d_sum_v, float* d_sum_friction) {
        lattice.build(n, d_X);
        compute_lattice_pwints<Pt, n_max, pw_int, pw_friction><<<(n + TILE_SIZE - 1)/TILE_SIZE, TILE_SIZE>>>(
            n, d_X, d_dX, d_old_v, d_sum_v, d_sum_friction, d_lattice);
    }
};

template<typename Pt, int n_max> using Lattice_solver = Order2_solver<Pt, n_max, Lattice_computer>;
