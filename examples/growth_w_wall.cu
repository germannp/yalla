// Simulate growing mesenchyme constrained by a planar wall
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <time.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.0f;
const auto mean_dist = 0.75f;
const auto r_protrusion = 1.0f;
const auto protrusion_strength = 0.15f;
const auto prots_per_cell = 1;
const auto n_0 = 500;
const auto n_max = 100000;

const auto dt = 0.1;
auto n_time_steps = 500;
auto skip_step = n_time_steps/100;

auto update_prob = 0.5;

auto prolif_rate = 0.005;

enum Cell_types { wall_node, mesenchyme };

__device__ Cell_types* d_type;

template<typename Pt>
__device__ float wall_friction(Pt Xi, Pt r, float dist, int i, int j)
{
    if (i == 0 or j == 0) return 0;
    if (dist < r_max) return 1;

    return 0;
}

__device__ float3 relu_force(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};

    // No one interacts with the wall node through pwints
    if(i==0 or j==0)
        return dF;

    if (i == j)
        return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) - fmaxf(dist - 0.8, 0);

    dF.x += r.x * F / dist;
    dF.y += r.y * F / dist;
    dF.z += r.z * F / dist;

    return dF;
}

__global__ void proliferate(float rate, int n_cells, curandState* d_state,
    float3* d_X, float3* d_old_v, int* d_n_cells)
{
    D_ASSERT(n_cells * rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;  // Dividing new cells is problematic!

    if(i == 0) return; // The wall node doesn't proliferate

    auto rnd = curand_uniform(&d_state[i]);
    if (rnd > rate) return;

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
    d_X[n].x = d_X[i].x + mean_dist / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_dist / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_dist / 4 * cosf(theta);
    d_type[n] = d_type[i];
    d_old_v[n] = d_old_v[i];
}

__global__ void update_protrusions_wall(const int n_cells,
    const Grid* __restrict__ d_grid, const float3* __restrict d_X,
    curandState* d_state, Link* d_link, float update_prob)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n_cells) * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rand_nb_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube] + 1;
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
    if (a==0) return;
    auto b =
        d_grid->d_point_id[d_grid->d_cube_start[rand_nb_cube] +
                           min(static_cast<int>(
                                   curand_uniform(&d_state[i]) * cells_in_cube),
                               cells_in_cube - 1)];

    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;
    if (b == 0) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto noise = curand_uniform(&d_state[i]);

    auto updated = noise < update_prob;

    if (not_initialized or updated){
        link->a = a;
        link->b = b;
    }
}

int main(int argc, const char* argv[])
{

    // Solution<float3, Grid_solver> cells{n_max};
    Solution<float3, Gabriel_solver> cells{n_max};

    *cells.h_n = n_0;

    cells.h_X[0].x = 0.0;
    cells.h_X[0].y = 0.0;
    cells.h_X[0].z = -mean_dist;

    random_sphere(0.5, cells, 1);

    for (auto i = 1; i < n_0; i++) {
        if(cells.h_X[i].z < 0.0f)
            cells.h_X[i].z *= -1.0f;
    }
    cells.copy_to_device();

    Property<Cell_types> type{n_max};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));

    auto wall = [&](const int n, const float3* __restrict__ d_X, float3* d_dX){
        return wall_forces<float3, xy_wall_relu_force>(n, d_X, d_dX, 0);
    };


    type.h_prop[0] = wall_node;
    for (auto i = 1; i < *cells.h_n; i++)
        type.h_prop[i] = mesenchyme;
    type.copy_to_device();
    cells.copy_to_device();

    // Initial relaxation
    for (auto time_step = 0; time_step <= 100; time_step++)
        cells.take_step<relu_force, friction_on_background>(dt, wall);

    Links protrusions{n_max, protrusion_strength};
    protrusions.set_d_n(n_0);

    auto interc_wall = [&](const int n, const float3* __restrict__ d_X, float3* d_dX){
        return link_wall_forces<float3, linear_force, xy_wall_relu_force>(protrusions, n, d_X, d_dX, 0);
    };

    Grid grid{n_max};

    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);


    // Simulate growth
    Vtk_output output{"growth_w_wall","output/", true};

    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        protrusions.set_d_n(cells.get_d_n() * prots_per_cell);
        grid.build(cells, r_protrusion);

        update_protrusions_wall<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            cells.get_d_n(), grid.d_grid, cells.d_X, protrusions.d_state,
            protrusions.d_link, update_prob);

        cells.take_step<relu_force, wall_friction>(dt, interc_wall);
        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            prolif_rate, cells.get_d_n(), d_state,
            cells.d_X, cells.d_old_v, cells.d_n);

        if(time_step % skip_step == 0){
            cudaDeviceSynchronize();
            cells.copy_to_host();
            protrusions.copy_to_host();
            type.copy_to_host();
            output.write_positions(cells);
            output.write_links(protrusions);
            output.write_property(type);
        }
    }

    return 0;
}
