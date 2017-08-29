// Solvers for N-body problems
#pragma once

#include <assert.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <functional>

#include "cudebug.cuh"


// Interactions must be specified between two points Xi and Xj, with  r = Xi -
// Xj. The type Pt (e.g. float3, see dtypes.cuh) contains the variables to be
// integrated, e.g. position or concentrations.
template<typename Pt>
using Pairwise_interaction = Pt(Pt Xi, Pt r, float dist, int i, int j);

// Similarly, a pairwise friction coefficient can be specified, see
// http://dx.doi.org/10.1007/s10237-014-0613-5. By default bolls closer
// than 1 exert friction on each other.
template<typename Pt>
using Pairwise_friction = float(Pt Xi, Pt r, float dist, int i, int j);

template<typename Pt>
__device__ float friction_w_neighbour(Pt Xi, Pt r, float dist, int i, int j)
{
    if (i == j) return 0;

    if (dist < 1) return 1;

    return 0;
}

__device__ float friction_on_background(
    float3 Xi, float3 r, float dist, int i, int j)
{
    return 0;
}

// In addition a generic force can be passed optionally:
template<typename Pt>
using Generic_forces =
    std::function<void(const Pt* __restrict__ d_X, Pt* d_dX)>;

template<typename Pt>
void no_gen_forces(const Pt* __restrict__ d_X, Pt* d_dX) {}

// Generic forces are computed before the pairwise interactions, e.g. to reset
// the number of neighbours between computations of the derivatives.


// Solution<Pt, n_max, Solver> combines a method, Solver, with a point type, Pt.
// It stores the variables on the host and specifies how the variables on the
// device can be accessed and how new steps are computed. However, all the GPU
// action is happening in the Solver classes.
template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution : public Solver<Pt, n_max> {
public:
    Pt* h_X = (Pt*)malloc(n_max * sizeof(Pt));  // Current variables on host
    Pt* d_X = Solver<Pt, n_max>::d_X;           // Variables on device (GPU)
    int* h_n = (int*)malloc(sizeof(int));       // Number of points
    int* d_n = Solver<Pt, n_max>::d_n;
    Solution(int n_0 = n_max) { *h_n = n_0; }
    void copy_to_device()
    {
        assert(*h_n <= n_max);
        cudaMemcpy(d_X, h_X, n_max * sizeof(Pt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void copy_to_host()
    {
        cudaMemcpy(h_X, d_X, n_max * sizeof(Pt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= n_max);
    }
    int get_d_n() { return Solver<Pt, n_max>::get_d_n(); }
    template<Pairwise_interaction<Pt> pw_int,
        Pairwise_friction<Pt> pw_friction = friction_w_neighbour<Pt>>
    void take_step(float dt, Generic_forces<Pt> gen_forces = no_gen_forces<Pt>)
    {
        return Solver<Pt, n_max>::template take_step<pw_int, pw_friction>(
            dt, gen_forces);
    }
};


// 2nd order solver for the equation v = F + <v(t - dt)> for x, y, and z, where
// <v> is the mean velocity of the neighbours weighted by the friction
// coefficients. The center of mass is kept fixed. Solves dw/dt = F_w for other
// variables in Pt.
template<typename Pt>
__global__ void euler_step(const int n, const float dt,
    const Pt* __restrict__ d_X0, const Pt mean_dX, Pt* d_dX, Pt* d_X)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX[i].x -= mean_dX.x;
    d_dX[i].y -= mean_dX.y;
    d_dX[i].z -= mean_dX.z;

    d_X[i] = d_X0[i] + d_dX[i] * dt;
}

template<typename Pt>
__global__ void heun_step(const int n, const float dt,
    const Pt* __restrict__ d_dX, const Pt mean_dX1, Pt* d_dX1, Pt* d_X,
    float3* d_old_v)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX1[i].x -= mean_dX1.x;
    d_dX1[i].y -= mean_dX1.y;
    d_dX1[i].z -= mean_dX1.z;

    d_X[i] += (d_dX[i] + d_dX1[i]) * 0.5 * dt;

    d_old_v[i].x = (d_dX[i].x + d_dX1[i].x) * 0.5;
    d_old_v[i].y = (d_dX[i].y + d_dX1[i].y) * 0.5;
    d_old_v[i].z = (d_dX[i].z + d_dX1[i].z) * 0.5;
}

template<typename Pt>
__global__ void add_rhs(const int n, const float3* __restrict__ d_sum_v,
    const float* __restrict__ d_sum_friction, Pt* d_dX)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    D_ASSERT(d_dX[i].x == d_dX[i].x);  // For NaN f != f
    D_ASSERT(d_sum_v[i].x == d_sum_v[i].x);

    if (d_sum_friction[i] > 0) {
        d_dX[i].x += d_sum_v[i].x / d_sum_friction[i];
        d_dX[i].y += d_sum_v[i].y / d_sum_friction[i];
        d_dX[i].z += d_sum_v[i].z / d_sum_friction[i];
    }
}

// Computer specifies how pairwise interactions are computed.
template<typename Pt, int n_max, template<typename, int> class Computer>
class Heun_solver : public Computer<Pt, n_max> {
protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    float3 *d_old_v, *d_sum_v;
    float* d_sum_friction;
    int* d_n;
    Heun_solver()
    {
        cudaMalloc(&d_X, n_max * sizeof(Pt));
        cudaMalloc(&d_dX, n_max * sizeof(Pt));
        cudaMalloc(&d_X1, n_max * sizeof(Pt));
        cudaMalloc(&d_dX1, n_max * sizeof(Pt));

        cudaMalloc(&d_old_v, n_max * sizeof(float3));
        thrust::fill(thrust::device, d_old_v, d_old_v + n_max, float3{0});
        cudaMalloc(&d_sum_v, n_max * sizeof(float3));

        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_sum_friction, n_max * sizeof(int));
    }
    int get_d_n()
    {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= n_max);
        return n;
    }
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void take_step(float dt, Generic_forces<Pt> gen_forces)
    {
        auto n = get_d_n();

        // 1st step
        thrust::fill(thrust::device, d_dX, d_dX + n, Pt{0});
        thrust::fill(thrust::device, d_sum_friction, d_sum_friction + n, 0);
        thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3{0});
        gen_forces(d_X, d_dX);
        Computer<Pt, n_max>::template pwints<pw_int, pw_friction>(
            n, d_X, d_dX, d_old_v, d_sum_v, d_sum_friction);
        add_rhs<<<(n + 32 - 1) / 32, 32>>>(
            n, d_sum_v, d_sum_friction, d_dX);  // ceil int div.
        auto mean_dX =
            thrust::reduce(thrust::device, d_dX, d_dX + n, Pt{0}) / n;
        euler_step<<<(n + 32 - 1) / 32, 32>>>(n, dt, d_X, mean_dX, d_dX, d_X1);

        // 2nd step
        thrust::fill(thrust::device, d_dX1, d_dX1 + n, Pt{0});
        thrust::fill(thrust::device, d_sum_friction, d_sum_friction + n, 0);
        thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3{0});
        gen_forces(d_X1, d_dX1);
        Computer<Pt, n_max>::template pwints<pw_int, pw_friction>(
            n, d_X1, d_dX1, d_old_v, d_sum_v, d_sum_friction);
        add_rhs<<<(n + 32 - 1) / 32, 32>>>(n, d_sum_v, d_sum_friction, d_dX1);
        auto mean_dX1 =
            thrust::reduce(thrust::device, d_dX1, d_dX1 + n, Pt{0}) / n;
        heun_step<<<(n + 32 - 1) / 32, 32>>>(
            n, dt, d_dX, mean_dX1, d_dX1, d_X, d_old_v);
    }
};


// Compute pairwise interactions and frictions one thread per point, to
// TILE_SIZE points at a time, after http://http.developer.nvidia.com/
// GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, Pairwise_interaction<Pt> pw_int,
    Pairwise_friction<Pt> pw_friction>
__global__ void compute_tile(const int n, const Pt* __restrict__ d_X, Pt* d_dX,
    const float3* __restrict__ d_old_v, float3* d_sum_v, float* d_sum_friction)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Pt shX[TILE_SIZE];

    Pt Xi{0};
    if (i < n) Xi = d_X[i];
    Pt F{0};
    float3 sum_v{0};
    float sum_friction = 0;
    for (auto tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        auto j = tile_start + threadIdx.x;
        if (j < n) {
            shX[threadIdx.x] = d_X[j];
        }
        __syncthreads();

        for (auto k = 0; k < TILE_SIZE; k++) {
            auto j = tile_start + k;
            if ((i < n) and (j < n)) {
                auto r = Xi - shX[k];
                auto dist = norm3df(r.x, r.y, r.z);
                F += pw_int(Xi, r, dist, i, j);
                auto friction = pw_friction(Xi, r, dist, i, j);
                sum_friction += friction;
                sum_v += friction * d_old_v[j];
            }
        }
    }

    if (i < n) {
        d_dX[i] += F;
        d_sum_friction[i] = sum_friction;
        d_sum_v[i] = sum_v;
    }
}

template<typename Pt, int n_max>
class Tile_computer {
protected:
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(const int n, Pt* d_X, Pt* d_dX,
        const float3* __restrict__ d_old_v, float3* d_sum_v,
        float* d_sum_friction)
    {
        compute_tile<Pt, pw_int, pw_friction>
            <<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
                n, d_X, d_dX, d_old_v, d_sum_v, d_sum_friction);
    }
};

template<typename Pt, int n_max>
using Tile_solver = Heun_solver<Pt, n_max, Tile_computer>;


// Compute pairwise interactions and frictions with sorting based grid ONLY for
// bolls closer than CUBE_SIZE. Scales linearly in n, faster with maybe 7k
// bolls. After http://developer.download.nvidia.com/compute/cuda/1.1-Beta/
// x86_website/projects/particles/doc/particles.pdf
const auto CUBE_SIZE = 1.f;
const auto GRID_SIZE = 50;
const auto N_CUBES = GRID_SIZE * GRID_SIZE * GRID_SIZE;

template<typename Pt>
__global__ void compute_cube_id(const int n, const float cube_size,
    const Pt* __restrict__ d_X, int* d_cube_id, int* d_point_id)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x / cube_size) + GRID_SIZE / 2) +
        (floor(d_X[i].y / cube_size) + GRID_SIZE / 2) * GRID_SIZE +
        (floor(d_X[i].z / cube_size) + GRID_SIZE / 2) * GRID_SIZE * GRID_SIZE);
    D_ASSERT(id >= 0);
    D_ASSERT(id < N_CUBES);
    d_cube_id[i] = id;
    d_point_id[i] = i;
}

__global__ void compute_cube_start_and_end(const int n,
    const int* __restrict__ d_cube_id, int* d_cube_start, int* d_cube_end)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto cube = d_cube_id[i];
    auto prev = i > 0 ? d_cube_id[i - 1] : -1;
    if (cube != prev) d_cube_start[cube] = i;
    auto next = i < n - 1 ? d_cube_id[i + 1] : d_cube_id[i] + 1;
    if (cube != next) d_cube_end[cube] = i;
}

template<int n_max>
class Grid {
public:
    int *d_cube_id, *d_point_id, *d_cube_start, *d_cube_end;
    Grid<n_max>* d_grid;
    Grid()
    {
        cudaMalloc(&d_cube_id, n_max * sizeof(int));
        cudaMalloc(&d_point_id, n_max * sizeof(int));
        cudaMalloc(&d_cube_start, N_CUBES * sizeof(int));
        cudaMalloc(&d_cube_end, N_CUBES * sizeof(int));

        cudaMalloc(&d_grid, sizeof(Grid<n_max>));
        cudaMemcpy(d_grid, this, sizeof(Grid<n_max>), cudaMemcpyHostToDevice);
    }
    template<typename Pt>
    void build(const int n, const Pt* __restrict__ d_X,
        const float cube_size = CUBE_SIZE)
    {
        compute_cube_id<<<(n + 32 - 1) / 32, 32>>>(
            n, cube_size, d_X, d_cube_id, d_point_id);
        thrust::fill(thrust::device, d_cube_start, d_cube_start + N_CUBES, -1);
        thrust::fill(thrust::device, d_cube_end, d_cube_end + N_CUBES, -2);
        thrust::sort_by_key(
            thrust::device, d_cube_id, d_cube_id + n, d_point_id);
        compute_cube_start_and_end<<<(n + 32 - 1) / 32, 32>>>(
            n, d_cube_id, d_cube_start, d_cube_end);
    }
    template<typename Pt, int n_max_solution,
        template<typename, int> class Solver>
    void build(Solution<Pt, n_max_solution, Solver>& bolls,
        const float cube_size = CUBE_SIZE)
    {
        auto n = bolls.get_d_n();
        assert(n <= n_max);
        build(n, bolls.d_X, cube_size);
    }
};


__constant__ int d_nhood[27];  // This is wasted w/o Grid_computer

template<typename Pt, int n_max, Pairwise_interaction<Pt> pw_int,
    Pairwise_friction<Pt> pw_friction>
__global__ void compute_cube(const int n, const Pt* __restrict__ d_X,
    const float3* __restrict__ d_old_v, const Grid<n_max>* __restrict__ d_grid,
    Pt* d_dX, float3* d_sum_v, float* d_sum_friction)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto Xi = d_X[d_grid->d_point_id[i]];
    Pt F{0};
    float3 sum_v{0};
    float sum_friction = 0;
    for (auto j = 0; j < 27; j++) {
        auto cube = d_grid->d_cube_id[i] + d_nhood[j];
        for (auto k = d_grid->d_cube_start[cube]; k <= d_grid->d_cube_end[cube];
             k++) {
            auto Xj = d_X[d_grid->d_point_id[k]];
            auto r = Xi - Xj;
            auto dist = norm3df(r.x, r.y, r.z);
            if (dist >= CUBE_SIZE) continue;

            F += pw_int(
                Xi, r, dist, d_grid->d_point_id[i], d_grid->d_point_id[k]);
            auto friction = pw_friction(
                Xi, r, dist, d_grid->d_point_id[i], d_grid->d_point_id[k]);
            sum_friction += friction;
            sum_v += friction * d_old_v[d_grid->d_point_id[k]];
        }
    }
    d_dX[d_grid->d_point_id[i]] += F;
    d_sum_v[d_grid->d_point_id[i]] = sum_v;
    d_sum_friction[d_grid->d_point_id[i]] = sum_friction;
}

template<typename Pt, int n_max>
class Grid_computer {
protected:
    Grid<n_max> grid;
    Grid_computer()
    {
        int h_nhood[27];
        h_nhood[0] = -1;
        h_nhood[1] = 0;
        h_nhood[2] = 1;
        for (auto i = 0; i < 3; i++) {
            h_nhood[i + 3] = h_nhood[i % 3] - GRID_SIZE;
            h_nhood[i + 6] = h_nhood[i % 3] + GRID_SIZE;
        }
        for (auto i = 0; i < 9; i++) {
            h_nhood[i + 9] = h_nhood[i % 9] - GRID_SIZE * GRID_SIZE;
            h_nhood[i + 18] = h_nhood[i % 9] + GRID_SIZE * GRID_SIZE;
        }
        cudaMemcpyToSymbol(d_nhood, &h_nhood, 27 * sizeof(int));
    }
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(int n, Pt* d_X, Pt* d_dX, const float3* __restrict__ d_old_v,
        float3* d_sum_v, float* d_sum_friction)
    {
        grid.build(n, d_X);
        compute_cube<Pt, n_max, pw_int, pw_friction>
            <<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
                n, d_X, d_old_v, grid.d_grid, d_dX, d_sum_v, d_sum_friction);
    }
};

template<typename Pt, int n_max>
using Grid_solver = Heun_solver<Pt, n_max, Grid_computer>;
