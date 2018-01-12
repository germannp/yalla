// Proliferating mesenchyme between two epithelial layers
#include <curand_kernel.h>
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


const auto n_0 = 3000;
const auto n_max = 61000;
const auto proliferation_rate = 0.01;
const auto r_max = 1.f;
const auto mean_distance = 0.75;
const auto prots_per_cell = 1;
const auto r_protrusion = 2;
const auto n_time_steps = 500;
const auto dt = 0.2;
enum Cell_types { mesoderm, mesenchyme, ectoderm, aer };

MAKE_PT(Lb_cell, w, f, theta, phi);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Lb_cell lb_force(Lb_cell Xi, Lb_cell r, float dist, int i, int j)
{
    Lb_cell dF{0};
    if (i == j) {
        dF.w = -0.2 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.2 * (d_type[i] == mesenchyme) * Xi.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    auto both_mesoderm = (d_type[i] == mesoderm) and (d_type[j] == mesoderm);
    auto both_mesenchyme =
        (d_type[i] == mesenchyme) and (d_type[j] == mesenchyme);
    auto both_ectoderm = (d_type[i] > mesenchyme) and (d_type[j] > mesenchyme);
    if (both_mesoderm or both_mesenchyme or both_ectoderm) {
        F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0) * 1.2;
    } else {
        F = fmaxf(0.8 - dist, 0) * 2 - fmaxf(dist - 0.9, 0) * 1.2;
    }
    dF.x = r.x * F / dist * (d_type[i] != mesoderm);
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;
    dF.w = -0.1 * r.w * (d_type[i] == mesenchyme);
    dF.f = -0.1 * r.f * (d_type[i] == mesenchyme);

    if (d_type[j] == mesenchyme) {
        d_mes_nbs[i] += 1;
        return dF;
    } else
        d_epi_nbs[i] += 1;

    if (not(both_mesoderm or both_ectoderm)) return dF;

    dF += rigidity_force(Xi, r, dist) * 0.1;
    return dF;
}


__global__ void update_protrusions(const int n_cells,
    const Grid* __restrict__ d_grid, const Lb_cell* __restrict d_X,
    curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto a = d_grid->d_point_id[j];
    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto initialized = link->a != 0 or link->b != 0;
    auto r = d_X[link->a] - d_X[link->b];
    auto dist = norm3df(r.x, r.y, r.z);
    if (initialized and ((dist < r_max) or (dist > r_protrusion))) {
        link->a = a;
        link->b = a;
    }

    auto rnd_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rnd_cube] - d_grid->d_cube_start[rnd_cube];
    if (cells_in_cube < 1) return;

    auto rnd_cell =
        min(static_cast<int>(curand_uniform(&d_state[i]) * cells_in_cube),
            cells_in_cube - 1);
    auto b = d_grid->d_point_id[d_grid->d_cube_start[rnd_cube] + rnd_cell];
    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if ((new_dist < r_max) or (new_dist > r_protrusion)) return;

    auto more_along_w = fabs(new_r.w / new_dist) > fabs(r.w / dist) + 0.01 * 0;
    // auto high_f = (d_X[a].f + d_X[b].f) > 0.01;
    auto high_f = false;
    if (not initialized or more_along_w or high_f) {
        link->a = a;
        link->b = b;
    }
}


__device__ int* d_clone;

__global__ void proliferate(
    int n_0, Lb_cell* d_X, int* d_n_cells, curandState* d_state)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_0) return;  // Dividing new cells is problematic!

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
    D_ASSERT(n <= n_max);
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
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
    d_clone[n] = d_clone[i];
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Lb_cell, Grid_solver> cells{n_max};
    *cells.h_n = n_0;
    random_disk(r_max / 2, cells);
    Property<Cell_types> type{n_max};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0 / 2; i++) {
        type.h_prop[i] = mesoderm;
        cells.h_X[i].y /= 1.5;
        cells.h_X[i].theta = -M_PI / 2;
        type.h_prop[i + n_0 / 2] = ectoderm;
        cells.h_X[i + n_0 / 2].x += mean_distance / 2;
        cells.h_X[i + n_0 / 2].w = 1;
        cells.h_X[i + n_0 / 2].y /= 1.5;
        cells.h_X[i + n_0 / 2].theta = M_PI / 2;
    }
    cells.copy_to_device();
    type.copy_to_device();
    Property<int> n_mes_nbs{n_max};
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<int> n_epi_nbs{n_max};
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    for (auto i = 0; i < 100; i++) cells.take_step<lb_force>(dt);
    cells.copy_to_host();
    for (auto i = n_0 / 2; i < n_0; i++) {
        if ((fabs(cells.h_X[i].y) < 0.5) and (fabs(cells.h_X[i].z) < 4)) {
            cells.h_X[i].f = 1;
            type.h_prop[i] = aer;
        }
    }
    *cells.h_n += 100;
    random_disk(mean_distance * 1.5, cells, n_0);
    for (auto i = n_0; i < *cells.h_n; i++) {
        cells.h_X[i].x = 0.1;
        cells.h_X[i].y /= 2.5;
        type.h_prop[i] = mesenchyme;
    }
    Property<int> clone{n_max, "clone"};
    cudaMemcpyToSymbol(d_clone, &clone.d_prop, sizeof(d_clone));
    for (auto i = 0; i < *cells.h_n; i++) clone.h_prop[i] = i;
    cells.copy_to_device();
    type.copy_to_device();
    clone.copy_to_device();
    Links protrusions(n_max * prots_per_cell, 0.1);
    protrusions.set_d_n(n_0 * prots_per_cell);
    auto intercalation = std::bind(link_forces<Lb_cell>, protrusions,
        std::placeholders::_1, std::placeholders::_2);
    Grid grid{n_max};

    // Proliferate
    Vtk_output output{"initialization"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        type.copy_to_host();
        clone.copy_to_host();
        protrusions.copy_to_host();

        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            cells.get_d_n(), cells.d_X, cells.d_n, protrusions.d_state);
        protrusions.set_d_n(cells.get_d_n() * prots_per_cell);
        grid.build(cells, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            cells.get_d_n(), grid.d_grid, cells.d_X, protrusions.d_state,
            protrusions.d_link);
        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);
        cells.take_step<lb_force>(dt, intercalation);

        output.write_positions(cells);
        output.write_links(protrusions);
        output.write_property(type);
        output.write_property(clone);
        // output.write_polarity(cells);
        output.write_field(cells, "Wnt");
        output.write_field(cells, "Fgf", &Lb_cell::f);
    }

    return 0;
}
