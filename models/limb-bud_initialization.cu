// Proliferating mesenchyme between two epithelial layers
#include <stdio.h>
#include <thread>
#include <functional>
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../include/cudebug.cuh"
#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
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
enum Cell_types {mesoderm, mesenchyme, ectoderm, aer};

MAKE_PT(Lb_cell, w, f, theta, phi);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Lb_cell lb_force(Lb_cell Xi, Lb_cell r, float dist, int i, int j) {
    Lb_cell dF {0};
    if (i == j) {
        dF.w = - 0.2*(d_type[i] == mesenchyme)*Xi.w;
        dF.f = - 0.2*(d_type[i] == mesenchyme)*Xi.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0);
    } else {
        F = fmaxf(0.8 - dist, 0)*2 - fmaxf(dist - 0.9, 0);
    }
    dF.x = r.x*F/dist*(d_type[i] != mesoderm);
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = - 0.1*r.w*(d_type[i] == mesenchyme);
    dF.f = - 0.1*r.f*(d_type[i] == mesenchyme);

    if (d_type[j] == mesenchyme) { d_mes_nbs[i] += 1; return dF; }
    else d_epi_nbs[i] += 1;

    if (d_type[i] != d_type[j]) return dF;

    dF += rigidity_force(Xi, r, dist)*0.1;
    return dF;
}


__global__ void update_protrusions(const int n_cells, const Grid<n_max>* __restrict__ d_grid,
        const Lb_cell* __restrict d_X, curandState* d_state, Link* d_link) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5)/prots_per_cell);
    auto rand_nb_cube = d_grid->d_cube_id[j]
        + d_moore_nhood[min(static_cast<int>(curand_uniform(&d_state[i])*27), 26)];
    auto cells_in_cube = d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
    auto b = d_grid->d_point_id[d_grid->d_cube_start[rand_nb_cube]
        + min(static_cast<int>(curand_uniform(&d_state[i])*cells_in_cube), cells_in_cube - 1)];
    D_ASSERT(a >= 0); D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0); D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a*prots_per_cell + i%prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto more_along_w = fabs(new_r.w/new_dist) > fabs(old_r.w/old_dist) + 0.01;
    auto high_f = (d_X[a].f + d_X[b].f) > 0.01;
    if (not_initialized or more_along_w or high_f) {
        link->a = a;
        link->b = b;
    }
}


__global__ void proliferate(int n_0, Lb_cell* d_X, int* d_n_cells, curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_0) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > proliferation_rate) return;
            break;
        }
        default:
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
    }

    auto n = atomicAdd(d_n_cells, 1);
    D_ASSERT(n <= n_max);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w/2;
        d_X[i].w = d_X[i].w/2;
        d_X[n].f = d_X[i].f/2;
        d_X[i].f = d_X[i].f/2;
    } else {
        d_X[n].w = d_X[i].w;
        d_X[n].f = d_X[i].f;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Lb_cell, n_max, Grid_solver> bolls(n_0);
    uniform_circle(r_max/2, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0/2; i++) {
        bolls.h_X[i].y /= 1.5;
        bolls.h_X[i].theta = -M_PI/2;
        type.h_prop[i] = mesoderm;
        bolls.h_X[i + n_0/2].x += mean_distance/2;
        bolls.h_X[i + n_0/2].w = 1;
        bolls.h_X[i + n_0/2].y /= 1.5;
        bolls.h_X[i + n_0/2].theta = M_PI/2;
        type.h_prop[i + n_0/2] = ectoderm;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    for (auto i = 0; i < 100; i++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<lb_force>(dt);
    }
    bolls.copy_to_host();
    *bolls.h_n += 100;
    uniform_circle(mean_distance*1.5, bolls, n_0);
    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = 0.1;
        bolls.h_X[i].y /= 1.5;
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Links<n_max*prots_per_cell> protrusions(0.1, n_0*prots_per_cell);
    auto intercalation = std::bind(
        link_forces<static_cast<int>(n_max*prots_per_cell), Lb_cell>,
        protrusions, std::placeholders::_1, std::placeholders::_2);
    Grid<n_max> grid;

    // Proliferate
    Vtk_output output("initialization");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        type.copy_to_host();
        protrusions.copy_to_host();
        proliferate<<<(bolls.get_d_n() + 128 - 1)/128, 128>>>(bolls.get_d_n(), bolls.d_X, bolls.d_n,
            protrusions.d_state);
        protrusions.set_d_n(bolls.get_d_n()*prots_per_cell);
        grid.build(bolls, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(bolls.get_d_n(),
            grid.d_grid, bolls.d_X, protrusions.d_state, protrusions.d_link);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<lb_force>(dt, intercalation);
        output.write_positions(bolls);
        output.write_links(protrusions);
        output.write_property(type);
        // output.write_polarity(bolls);
        output.write_field(bolls, "Wnt");
    }

    return 0;
}
