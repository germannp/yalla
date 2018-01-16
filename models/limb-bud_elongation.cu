// Simulate elongation of semisphere
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <functional>
#include <thread>

#include "../include/cudebug.cuh"
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto n_0 = 15000;
const auto n_max = 65000;
const auto r_max = 1.f;
const auto r_protrusion = 2.f;
const auto protrusion_strength = 0.1f;
const auto prots_per_cell = 1;
const auto n_time_steps = 200;
const auto skip_steps = 0;
const auto proliferation_rate = 1.386 / n_time_steps;  // log(4) = 1.386
const auto dt = 0.2f;
enum Cell_types { mesenchyme, epithelium, aer };

MAKE_PT(Lb_cell, w, f, theta, phi);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Lb_cell lb_force(Lb_cell Xi, Lb_cell r, float dist, int i, int j)
{
    Lb_cell dF{0};
    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = -0.2 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.2 * (d_type[i] == mesenchyme) * Xi.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    } else {
        F = fmaxf(0.8 - dist, 0) * 2 - fmaxf(dist - 0.9, 0);
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;
    dF.w = -0.1 * r.w * (d_type[i] == mesenchyme);
    dF.f = -0.1 * r.f * (d_type[i] == mesenchyme);

    if (d_type[j] == mesenchyme)
        d_mes_nbs[i] += 1;
    else
        d_epi_nbs[i] += 1;

    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += rigidity_force(Xi, r, dist) * 0.2;
    return dF;
}


__global__ void update_protrusions(const int n_cells,
    const Grid* __restrict__ d_grid, const Lb_cell* __restrict d_X,
    curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rnd_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rnd_cube] - d_grid->d_cube_start[rnd_cube];
    if (cells_in_cube < 1) return;

    auto rnd_cell =
        min(static_cast<int>(curand_uniform(&d_state[i]) * cells_in_cube),
            cells_in_cube - 1);
    auto a = d_grid->d_point_id[j];
    auto b = d_grid->d_point_id[d_grid->d_cube_start[rnd_cube] + rnd_cell];
    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto more_along_w =
        fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) + 0.01;
    auto high_f = (d_X[a].f + d_X[b].f) > 0.02;
    if (not_initialized or more_along_w or high_f) {
        link->a = a;
        link->b = b;
    }
}


__global__ void proliferate(
    float mean_distance, Lb_cell* d_X, int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * proliferation_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - proliferation_rate))
        return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto rnd = curand_uniform(&d_state[i]);
            if (rnd > proliferation_rate) return;
            break;
        }
        default:
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w / 2;
        d_X[i].w = d_X[i].w / 2;
        d_X[n].f = d_X[i].f / 2;
        d_X[i].f = d_X[i].f / 2;
    } else {
        d_X[n].w = d_X[i].w;
        d_X[n].f = d_X[i].f;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Lb_cell, Grid_solver> cells{n_max};
    *cells.h_n = n_0;
    random_sphere(0.733333, cells);
    Property<Cell_types> type{n_max};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        cells.h_X[i].x = fabs(cells.h_X[i].x);
        cells.h_X[i].y = cells.h_X[i].y / 1.5;
        type.h_prop[i] = mesenchyme;
    }
    cells.copy_to_device();
    type.copy_to_device();
    Property<int> n_mes_nbs{n_max};
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<int> n_epi_nbs{n_max};
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    Links protrusions(n_max * prots_per_cell, protrusion_strength);
    protrusions.set_d_n(n_0 * prots_per_cell);
    auto intercalation = std::bind(link_forces<Lb_cell>, protrusions,
        std::placeholders::_1, std::placeholders::_2);

    // Relax
    Grid grid{n_max};
    for (auto time_step = 0; time_step <= 100; time_step++) {
        grid.build(cells, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            cells.get_d_n(), grid.d_grid, cells.d_X, protrusions.d_state,
            protrusions.d_link);
        thrust::fill(
            thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        cells.take_step<lb_force>(dt, intercalation);
    }

    // Find epithelium
    cells.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 16 * 2 and
            cells.h_X[i].x > 0) {  // *2 for 2nd order solver
            cells.h_X[i].w = 1;
            if (fabs(cells.h_X[i].y) < 0.75 and cells.h_X[i].x > 5) {
                type.h_prop[i] = aer;
                cells.h_X[i].f = 1;
            } else {
                type.h_prop[i] = epithelium;
            }
            auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                              cells.h_X[i].y * cells.h_X[i].y +
                              cells.h_X[i].z * cells.h_X[i].z);
            cells.h_X[i].theta = acosf(cells.h_X[i].z / dist);
            cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x);
        }
    }
    cells.copy_to_device();
    type.copy_to_device();
    protrusions.reset([&](int a, int b) {
        return ((type.h_prop[a] > mesenchyme) or (type.h_prop[b] > mesenchyme));
    });
    // Relax epithelium before proliferate
    for (auto time_step = 0; time_step <= 50; time_step++)
        cells.take_step<lb_force>(dt, intercalation);

    // Simulate diffusion & intercalation
    Vtk_output output{"elongation"};
    for (auto time_step = 0; time_step <= n_time_steps / (skip_steps + 1);
         time_step++) {
        cells.copy_to_host();
        protrusions.copy_to_host();
        type.copy_to_host();

        std::thread calculation([&] {
            for (auto i = 0; i <= skip_steps; i++) {
                proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    0.733333, cells.d_X, cells.d_n, protrusions.d_state);
                protrusions.set_d_n(cells.get_d_n() * prots_per_cell);
                grid.build(cells, r_protrusion);
                update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32,
                    32>>>(cells.get_d_n(), grid.d_grid, cells.d_X,
                    protrusions.d_state, protrusions.d_link);
                thrust::fill(thrust::device, n_mes_nbs.d_prop,
                    n_mes_nbs.d_prop + cells.get_d_n(), 0);
                thrust::fill(thrust::device, n_epi_nbs.d_prop,
                    n_epi_nbs.d_prop + cells.get_d_n(), 0);
                cells.take_step<lb_force>(dt, intercalation);
            }
        });

        output.write_positions(cells);
        output.write_links(protrusions);
        output.write_property(type);
        // output.write_polarity(cells);
        output.write_field(cells, "Wnt");
        output.write_field(cells, "Fgf", &Lb_cell::f);

        calculation.join();
    }

    return 0;
}
