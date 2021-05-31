// Solvers for N-body problems
#pragma once

#include <assert.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <functional>

#include "cudebug.cuh"
#include "dtypes.cuh"


// Interactions must be specified between two points Xi and Xj, with  r = Xi -
// Xj. The type Pt (e.g. float3, see dtypes.cuh) contains the variables to be
// integrated, e.g. position or concentrations.
template<typename Pt>
using Pairwise_interaction = Pt(Pt Xi, Pt r, float dist, int i, int j);

// Similarly, a pairwise friction coefficient can be specified, see
// http://dx.doi.org/10.1007/s10237-014-0613-5. By default points closer
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

template<typename Pt>
__device__ float friction_on_background(Pt Xi, Pt r, float dist, int i, int j)
{
    return 0;
}

// In addition a generic force can be passed optionally:
template<typename Pt>
using Generic_forces =
    std::function<void(const int n, const Pt* __restrict__ d_X, Pt* d_dX)>;

template<typename Pt>
void no_gen_forces(const int n, const Pt* __restrict__ d_X, Pt* d_dX)
{}

// Generic forces are computed before the pairwise interactions, e.g. to reset
// the number of neighbours between computations of the derivatives.


// Solution<Pt, Solver> combines a method, Solver, with a point type, Pt. It
// stores the variables on the host and specifies how the variables on the
// device can be accessed and how new steps are computed. However, all the GPU
// action is happening in the Solver classes.
template<typename Pt, template<typename> class Solver>
class Solution : public Solver<Pt> {
public:
    Pt* h_X;                                      // Current variables on host
    Pt* const d_X = Solver<Pt>::d_X;              // Variables on device (GPU)
    float3* const d_old_v = Solver<Pt>::d_old_v;  // Velocities from previous step
    int* const h_n = (int*)malloc(sizeof(int));   // Number of points
    int* const d_n = Solver<Pt>::d_n;
    const int n_max;
    template<typename... Args>
    Solution(int n_max, Args... args) : n_max{n_max}, Solver<Pt>{n_max, args...}
    {
        *h_n = n_max;
        h_X = (Pt*)malloc(n_max * sizeof(Pt));
    }
    ~Solution()
    {
        free(h_X);
        free(h_n);
    }
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
    int get_d_n() { return Solver<Pt>::get_d_n(); }
    // Default template parameter for Pairwise_friction won't compile in CUDA 10.1
    template<Pairwise_interaction<Pt> pw_int>
    void take_step(float dt, Generic_forces<Pt> gen_forces = no_gen_forces<Pt>)
    {
        return Solver<Pt>::template take_step<pw_int, friction_w_neighbour>(
            dt, gen_forces);
    }
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void take_step(float dt, Generic_forces<Pt> gen_forces = no_gen_forces<Pt>)
    {
        return Solver<Pt>::template take_step<pw_int, pw_friction>(
            dt, gen_forces);
    }
};


// 2nd order solver for the equation v = F + <v(t - dt)> for x, y, and z, where
// <v> is the mean velocity of the neighbours weighted by the friction
// coefficients. One point or the center of mass needs is kept fix. Solves
// dw/dt = F_w for other variables in Pt.
template<typename Pt>
__global__ void euler_step(const int n, const float dt,
    const Pt* __restrict__ d_X0, const Pt fix_dX, Pt* d_dX, Pt* d_X)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX[i].x -= fix_dX.x;
    d_dX[i].y -= fix_dX.y;
    d_dX[i].z -= fix_dX.z;

    d_X[i] = d_X0[i] + d_dX[i] * dt;
}

template<typename Pt>
__global__ void heun_step(const int n, const float dt,
    const Pt* __restrict__ d_dX, const Pt fix_dX1, Pt* d_dX1, Pt* d_X,
    float3* d_old_v)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_dX1[i].x -= fix_dX1.x;
    d_dX1[i].y -= fix_dX1.y;
    d_dX1[i].z -= fix_dX1.z;

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
template<typename Pt, template<typename> class Computer>
class Heun_solver : public Computer<Pt> {
public:
    template<typename... Args>
    Heun_solver(int n_max, Args... args)
        : n_max{n_max}, Computer<Pt>{n_max, args...}
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
    ~Heun_solver()
    {
        cudaFree(d_X);
        cudaFree(d_dX);
        cudaFree(d_X1);
        cudaFree(d_dX1);

        cudaFree(d_old_v);
        cudaFree(d_sum_v);

        cudaFree(d_n);
        cudaFree(d_sum_friction);
    }
    void set_fixed() { fix_com = true; }
    void set_fixed(int point_id)
    {
        fix_com = false;
        fix_point = point_id;
    }

    void set_fixed_xy(int point_id)
    {
        fix_com = false;
        fix_com_z = true;
        fix_point = point_id;
    }

protected:
    Pt *d_X, *d_dX, *d_X1, *d_dX1;
    float3 *d_old_v, *d_sum_v;
    float* d_sum_friction;
    int* d_n;
    bool fix_com = true;
    bool fix_com_z = false;
    int fix_point;
    const int n_max;
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
        gen_forces(n, d_X, d_dX);
        Computer<Pt>::template pwints<pw_int, pw_friction>(
            n, d_X, d_old_v, d_dX, d_sum_v, d_sum_friction);
        add_rhs<<<(n + 32 - 1) / 32, 32>>>(
            n, d_sum_v, d_sum_friction, d_dX);  // ceil int div.
        Pt fix_dX;
        if (fix_com or fix_com_z) {
            fix_dX = thrust::reduce(thrust::device, d_dX, d_dX + n, Pt{0}) / n;
            if(fix_com_z){
                Pt temp;
                cudaMemcpy(
                    &temp, &d_dX[fix_point], sizeof(Pt), cudaMemcpyDeviceToHost);
                fix_dX.x = temp.x;
                fix_dX.y = temp.y;
            }
        } else {  // fix_com and fix_com_z are false
            cudaMemcpy(
                &fix_dX, &d_dX[fix_point], sizeof(Pt), cudaMemcpyDeviceToHost);
        }

        euler_step<<<(n + 32 - 1) / 32, 32>>>(n, dt, d_X, fix_dX, d_dX, d_X1);

        // 2nd step
        thrust::fill(thrust::device, d_dX1, d_dX1 + n, Pt{0});
        thrust::fill(thrust::device, d_sum_friction, d_sum_friction + n, 0);
        thrust::fill(thrust::device, d_sum_v, d_sum_v + n, float3{0});
        gen_forces(n, d_X1, d_dX1);
        Computer<Pt>::template pwints<pw_int, pw_friction>(
            n, d_X1, d_old_v, d_dX1, d_sum_v, d_sum_friction);
        add_rhs<<<(n + 32 - 1) / 32, 32>>>(n, d_sum_v, d_sum_friction, d_dX1);
        Pt fix_dX1;
        if (fix_com) {
            fix_dX1 =
                thrust::reduce(thrust::device, d_dX1, d_dX1 + n, Pt{0}) / n;
        } else {
            cudaMemcpy(&fix_dX1, &d_dX1[fix_point], sizeof(Pt),
                cudaMemcpyDeviceToHost);
        }
        heun_step<<<(n + 32 - 1) / 32, 32>>>(
            n, dt, d_dX, fix_dX1, d_dX1, d_X, d_old_v);
    }
};


// Compute pairwise interactions and frictions one thread per point, to
// TILE_SIZE points at a time, after http://http.developer.nvidia.com/
// GPUGems3/gpugems3_ch31.html.
const auto TILE_SIZE = 32;

template<typename Pt, Pairwise_interaction<Pt> pw_int,
    Pairwise_friction<Pt> pw_friction>
__global__ void compute_tile(const int n, const Pt* __restrict__ d_X,
    const float3* __restrict__ d_old_v, Pt* d_dX, float3* d_sum_v,
    float* d_sum_friction)
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
        if (j < n) { shX[threadIdx.x] = d_X[j]; }
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

template<typename Pt>
class Tile_computer {
public:
    Tile_computer(int n_max) {}

protected:
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(const int n, const Pt* __restrict__ d_X,
        const float3* __restrict__ d_old_v, Pt* d_dX, float3* d_sum_v,
        float* d_sum_friction)
    {
        compute_tile<Pt, pw_int, pw_friction>
            <<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
                n, d_X, d_old_v, d_dX, d_sum_v, d_sum_friction);
    }
};

template<typename Pt>
using Tile_solver = Heun_solver<Pt, Tile_computer>;


// Compute pairwise interactions and frictions with sorting based grid ONLY for
// points closer than cube_size. Scales linearly in n, faster with maybe 7k
// points. After http://developer.download.nvidia.com/compute/cuda/1.1-Beta/
// x86_website/projects/particles/doc/particles.pdf
template<typename Pt>
__global__ void compute_cube_id(const int n, const float cube_size,
    const int grid_size, const int n_cubes, const Pt* __restrict__ d_X,
    int* d_cube_id, int* d_point_id)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto id = static_cast<int>(
        (floor(d_X[i].x / cube_size) + grid_size / 2) +
        (floor(d_X[i].y / cube_size) + grid_size / 2) * grid_size +
        (floor(d_X[i].z / cube_size) + grid_size / 2) * grid_size * grid_size);
    D_ASSERT(id >= 0);
    D_ASSERT(id < n_cubes);
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

class Grid {
public:
    int *d_cube_id, *d_point_id, *d_cube_start, *d_cube_end;
    Grid* d_grid;
    const int n_max, grid_size, n_cubes;
    Grid(int n_max, int gs = 50)
        : n_max{n_max}, grid_size{gs}, n_cubes{gs * gs * gs}
    {
        cudaMalloc(&d_cube_id, n_max * sizeof(int));
        cudaMalloc(&d_point_id, n_max * sizeof(int));
        cudaMalloc(&d_cube_start, n_cubes * sizeof(int));
        cudaMalloc(&d_cube_end, n_cubes * sizeof(int));

        cudaMalloc(&d_grid, sizeof(Grid));
        cudaMemcpy(d_grid, this, sizeof(Grid), cudaMemcpyHostToDevice);
    }
    ~Grid()
    {
        cudaFree(d_cube_id);
        cudaFree(d_point_id);
        cudaFree(d_cube_start);
        cudaFree(d_cube_end);

        cudaFree(d_grid);
    }
    template<typename Pt>
    void build(
        const int n, const Pt* __restrict__ d_X, const float cube_size = 1)
    {
        compute_cube_id<<<(n + 32 - 1) / 32, 32>>>(
            n, cube_size, grid_size, n_cubes, d_X, d_cube_id, d_point_id);
        thrust::fill(thrust::device, d_cube_start, d_cube_start + n_cubes, -1);
        thrust::fill(thrust::device, d_cube_end, d_cube_end + n_cubes, -2);
        thrust::sort_by_key(
            thrust::device, d_cube_id, d_cube_id + n, d_point_id);
        compute_cube_start_and_end<<<(n + 32 - 1) / 32, 32>>>(
            n, d_cube_id, d_cube_start, d_cube_end);
    }
    template<typename Pt, template<typename> class Solver>
    void build(Solution<Pt, Solver>& points, const float cube_size = 1)
    {
        auto n = points.get_d_n();
        assert(n <= n_max);
        build(n, points.d_X, cube_size);
    }
};


__constant__ int d_nhood[27];  // This is wasted w/o Grid_computer

template<typename Pt, Pairwise_interaction<Pt> pw_int,
    Pairwise_friction<Pt> pw_friction>
__global__ void compute_cube(const int n, const Pt* __restrict__ d_X,
    const float3* __restrict__ d_old_v, const Grid* __restrict__ d_grid,
    const float cube_size, Pt* d_dX, float3* d_sum_v, float* d_sum_friction)
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
            if (dist >= cube_size) continue;

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

template<typename Pt>
class Grid_computer {
public:
    float cube_size;
    Grid_computer(int n_max, int grid_size = 50, float cube_size = 1)
        : grid{n_max, grid_size}, cube_size{cube_size}
    {
        int h_nhood[27];
        h_nhood[0] = -1;
        h_nhood[1] = 0;
        h_nhood[2] = 1;
        for (auto i = 0; i < 3; i++) {
            h_nhood[i + 3] = h_nhood[i % 3] - grid_size;
            h_nhood[i + 6] = h_nhood[i % 3] + grid_size;
        }
        for (auto i = 0; i < 9; i++) {
            h_nhood[i + 9] = h_nhood[i % 9] - grid_size * grid_size;
            h_nhood[i + 18] = h_nhood[i % 9] + grid_size * grid_size;
        }
        cudaMemcpyToSymbol(d_nhood, &h_nhood, 27 * sizeof(int));
    }

protected:
    Grid grid;
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(int n, const Pt* __restrict__ d_X,
        const float3* __restrict__ d_old_v, Pt* d_dX, float3* d_sum_v,
        float* d_sum_friction)
    {
        grid.build(n, d_X, cube_size);
        compute_cube<Pt, pw_int, pw_friction>
            <<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(n, d_X, d_old_v,
                grid.d_grid, cube_size, d_dX, d_sum_v, d_sum_friction);
    }
};

template<typename Pt>
using Grid_solver = Heun_solver<Pt, Grid_computer>;


// Compute pairwise interactions and frictions based on the grid solver,
// plus further refinement of the cell neighbourhood using the Gabriel
// method (Delile et al. 2017. Nature Communications,
// Marin-Riera et al. 2016, Bioinformatics).
template<typename Pt, Pairwise_interaction<Pt> pw_int,
    Pairwise_friction<Pt> pw_friction>
__global__ void compute_cube_gabriel(const int n, const Pt* __restrict__ d_X,
    const float3* __restrict__ d_old_v, const Grid* __restrict__ d_grid,
    const float cube_size, Pt* d_dX, float3* d_sum_v, float* d_sum_friction,
    const float gabriel_coefficient)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto id_i = d_grid->d_point_id[i];
    auto Xi = d_X[id_i];
    Pt F{0};
    float3 sum_v{0};
    float sum_friction = 0;

    int neighbour_id[100];
    float neighbour_dist[100];
    int n_neighs = 0;

    // First loop. Register all possible pairwise interactions based on
    // extensive sweep of the nearby cubes.
    for (auto j = 0; j < 27; j++) {
        auto cube = d_grid->d_cube_id[i] + d_nhood[j];
        for (auto k = d_grid->d_cube_start[cube]; k <= d_grid->d_cube_end[cube];
             k++) {
            auto j = d_grid->d_point_id[k];
            auto Xj = d_X[j];
            auto r = Xi - Xj;
            auto dist = norm3df(r.x, r.y, r.z);
            if (dist >= cube_size) continue;

            neighbour_id[n_neighs] = j;
            neighbour_dist[n_neighs] = dist;
            n_neighs++;
        }
    }

    // For an efficient implementation of the Gabriel method we sort the list
    // of possible interactions by distance
    for (auto m = 0; m < n_neighs - 1; m++) {
        auto min_val = neighbour_dist[m];
        auto min_index = m;
        for (auto n = m + 1; n < n_neighs; n++) {
            auto compare_val = neighbour_dist[n];
            if (compare_val < min_val) {
                min_index = n;
                min_val = compare_val;
            }
        }
        if (min_index != m) {
            auto id_temp = neighbour_id[min_index];
            neighbour_id[min_index] = neighbour_id[m];
            neighbour_id[m] = id_temp;
            neighbour_dist[min_index] = neighbour_dist[m];
            neighbour_dist[m] = min_val;
        }
    }

    // Second loop. We apply the Gabriel method by checking, for each pairwise
    // interaction if any other close cell falls inside the sphere
    // circumscribed by cells i and j. If that is the case, then pairwise
    // intereaction i-j is not considered for force calculations.
    for (auto m = n_neighs -1 ; m >= 0; m--) {
        auto gabriel_condition = true;
        auto j = neighbour_id[m];
        auto Xj = d_X[j];
        auto dist = neighbour_dist[m];
        if (j != id_i) {
            auto gabriel_radius = 0.5f * neighbour_dist[m] * gabriel_coefficient;
            auto mid_point = 0.5f * (Xi + Xj);
            for (auto n = m - 1 ; n >= 0; n--) {
                auto k = neighbour_id[n];
                auto r_mk = mid_point - d_X[k];
                auto dist_mk = norm3df(r_mk.x, r_mk.y, r_mk.z);
                if (dist_mk < gabriel_radius) {
                    gabriel_condition = false;
                    break;
                }
            }
        }
        if (gabriel_condition) {
            auto r = Xi - Xj;
            F += pw_int(Xi, r, dist, id_i, j);
            auto friction = pw_friction(Xi, r, dist, id_i, j);
            sum_friction += friction;
            sum_v += friction * d_old_v[j];
        }
    }

    d_dX[d_grid->d_point_id[i]] += F;
    d_sum_v[d_grid->d_point_id[i]] = sum_v;
    d_sum_friction[d_grid->d_point_id[i]] = sum_friction;
}

template<typename Pt>
class Gabriel_computer {
public:
    float cube_size;
    float gabriel_coefficient;
    Gabriel_computer(int n_max, int grid_size = 50, float cube_size = 1,
        float gabriel_coefficient = 0.8)
        : grid{n_max, grid_size}, cube_size{cube_size},
        gabriel_coefficient{gabriel_coefficient}
    {
        int h_nhood[27];
        h_nhood[0] = -1;
        h_nhood[1] = 0;
        h_nhood[2] = 1;
        for (auto i = 0; i < 3; i++) {
            h_nhood[i + 3] = h_nhood[i % 3] - grid_size;
            h_nhood[i + 6] = h_nhood[i % 3] + grid_size;
        }
        for (auto i = 0; i < 9; i++) {
            h_nhood[i + 9] = h_nhood[i % 9] - grid_size * grid_size;
            h_nhood[i + 18] = h_nhood[i % 9] + grid_size * grid_size;
        }
        cudaMemcpyToSymbol(d_nhood, &h_nhood, 27 * sizeof(int));
    }

protected:
    Grid grid;
    template<Pairwise_interaction<Pt> pw_int, Pairwise_friction<Pt> pw_friction>
    void pwints(int n, const Pt* __restrict__ d_X,
        const float3* __restrict__ d_old_v, Pt* d_dX, float3* d_sum_v,
        float* d_sum_friction)
    {
        grid.build(n, d_X, cube_size);
        compute_cube_gabriel<Pt, pw_int, pw_friction>
            <<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(n, d_X, d_old_v,
                grid.d_grid, cube_size, d_dX, d_sum_v, d_sum_friction, gabriel_coefficient);
    }
};

template<typename Pt>
using Gabriel_solver = Heun_solver<Pt, Gabriel_computer>;
