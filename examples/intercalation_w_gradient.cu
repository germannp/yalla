#include <curand_kernel.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/meix.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1f;
const auto n_max = 150000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;
float mean_proliferation_rate = 0.015f;
int n_time_steps = 500;

enum Cell_types { mesenchyme, epithelium };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;

MAKE_PT(Cell, w, f, theta, phi);

__device__ Cell force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        if (Xi.w < 0.f) Xi.w = 0.f;
        dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        if (Xi.f < 0.f) Xi.f = 0.f;

        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 2.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.1f;
    dF.f = -r.f * (d_type[i] == mesenchyme) * 0.1f;

    if (d_type[j] == epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    if (Xi.w < 0.f) Xi.w = 0.f;
    if (Xi.f < 0.f) Xi.f = 0.f;
    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += rigidity_force(Xi, r, dist) * 0.15;
    return dF;
}

__global__ void proliferate(float mean_rate, float mean_distance, Cell* d_X,
    int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * mean_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - mean_rate))
        return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            return;
        }
        case epithelium: {
            if (d_epi_nbs[i] > 14) return;
            if (d_mes_nbs[i] < 1) return;
            auto rnd = curand_uniform(&d_state[i]);
            if (rnd > mean_rate) return;

        }
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
    } else {
        d_X[n].w = d_X[i].w;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}

__global__ void update_protrusions(const int n_cells,
    const Grid<n_max>* __restrict__ d_grid, const Cell* __restrict d_X,
    curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rand_nb_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
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

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];

    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto not_initialized = link->a == link->b;
    auto noise = curand_uniform(&d_state[i]);
    auto superficial = d_X[a].w + d_X[b].w > 0.3f;  //sort cells close to the w
    auto parallel_to_w_gradient = false;            //source
    auto normal_to_f_gradient = false;
    if (superficial) {  //cells close to the w source respond to the f gradient
        normal_to_f_gradient =
            fabs(new_r.f / new_dist) < fabs(old_r.f / old_dist) * (1.f - noise);
    } else {            //cells close to the f source respond to the w gradient
        parallel_to_w_gradient =
            fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) * (1.f - noise);
    }

    if (not_initialized or parallel_to_w_gradient or normal_to_f_gradient) {
        link->a = a;
        link->b = b;
    }
}


int main(int argc, char const* argv[])
{
    std::string ic_file_name = "examples/sphere_ic.vtk";

    // Load the initial conditions
    Vtk_input input(ic_file_name);
    int n0 = input.n_bolls;
    Solution<Cell, n_max, Grid_solver> sphere(n0);

    input.read_positions(sphere);
    input.read_polarity(sphere);

    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    Property<n_max, int> intype;

    input.read_property(intype, "cell_type");  // we read it as an int, then we
                                               //translate to enum "Cell_types"

    for (int i = 0 ; i < n0 ; i++) {
        sphere.h_X[i].w = 0.0f;
        if (intype.h_prop[i] == 0) {
            type.h_prop[i] = mesenchyme;
        } else if (intype.h_prop[i] == 1) {
            type.h_prop[i] = epithelium;
            if (sphere.h_X[i].z > 0.0f){
                sphere.h_X[i].w = 1.0f;
                if (sphere.h_X[i].x > 0.0f and abs(sphere.h_X[i].y) < 2.5f and
                    sphere.h_X[i].z < 3.0f)
                    sphere.h_X[i].f = 1.0f;
            }

        }
    }

    sphere.copy_to_device();
    type.copy_to_device();

    Property<n_max, int> n_mes_nbs("n_mes_nbs");
    Property<n_max, int> n_epi_nbs("n_epi_nbs");
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // Declaration of links
    Links<static_cast<int>(n_max * prots_per_cell)> protrusions(
        protrusion_strength, n0 * prots_per_cell);
    auto intercalation =
    std::bind(link_forces<static_cast<int>(n_max * prots_per_cell), Cell>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    Grid<n_max> grid;

    // State for links
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(
        n_max, seed, d_state);

    Vtk_output output("intercalation_w_gradient", true);
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        sphere.copy_to_host();
        protrusions.copy_to_host();
        type.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + sphere.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + sphere.get_d_n(), 0);

        protrusions.set_d_n(sphere.get_d_n() * prots_per_cell);
        grid.build(sphere, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            sphere.get_d_n(), grid.d_grid, sphere.d_X,
            protrusions.d_state, protrusions.d_link);

        sphere.take_step<force>(dt, intercalation);

        proliferate<<<(sphere.get_d_n() + 128 - 1) / 128, 128>>>(
            mean_proliferation_rate, r_min, sphere.d_X,
            sphere.d_n, d_state);

        output.write_positions(sphere);
        output.write_links(protrusions);
        output.write_polarity(sphere);
        output.write_property(type);
        output.write_field(sphere);
        output.write_field(sphere, "f", &Cell::f);
    }

    return 0;
}
