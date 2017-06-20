// Simulate elongation of Xenopus aggregates, see Green (2014) Dev Dyn.
#include <math.h>
#include <stdio.h>
#include <thread>
#include <functional>
#include <curand_kernel.h>

#include "../include/cudebug.cuh"
#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/vtk.cuh"


const auto n_cells = 500;
const auto r_protrusion = 1.5f;
const auto prots_per_cell = 1;
const auto n_time_steps = 500;
const auto dt = 0.2f;


__device__ Po_cell lb_force(Po_cell Xi, Po_cell r, float dist, int i, int j) {
    Po_cell dF {0};
    if (i == j) return dF;

    if (dist > 1) return dF;

    auto F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0)/2;
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    return dF;
}


__device__ void protrusion_force(const Po_cell* __restrict__ d_X, const int a, const int b,
        const float strength, Po_cell* d_dX) {
    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -strength*r.x/dist);
    atomicAdd(&d_dX[a].y, -strength*r.y/dist);
    atomicAdd(&d_dX[a].z, -strength*r.z/dist);
    atomicAdd(&d_dX[b].x, strength*r.x/dist);
    atomicAdd(&d_dX[b].y, strength*r.y/dist);
    atomicAdd(&d_dX[b].z, strength*r.z/dist);

    Polarity r_hat {acosf(-r.z/dist), atan2(-r.y, -r.x)};
    auto Fa = pcp_force(d_X[a], r_hat);
    atomicAdd(&d_dX[a].theta, strength*Fa.theta);
    atomicAdd(&d_dX[a].phi, strength*Fa.phi);

    // r_hat.theta = acosf(r.z/dist);
    // r_hat.phi = atan2(r.y, r.x);
    auto Fb = pcp_force(d_X[b], r_hat);
    atomicAdd(&d_dX[b].theta, strength*Fb.theta);
    atomicAdd(&d_dX[b].phi, strength*Fb.phi);
}


__global__ void update_protrusions(const Grid<n_cells>* __restrict__ d_grid,
        const Po_cell* __restrict d_X, Link* d_link, curandState* d_state) {
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

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a*prots_per_cell + i%prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    Polarity old_rhat {acosf(-old_r.z/old_dist), atan2(-old_r.y, -old_r.x)};
    auto old_pcp = pol_scalar_product(d_X[a], old_rhat);
    Polarity new_rhat {acosf(-new_r.z/new_dist), atan2(-new_r.y, -new_r.x)};
    auto new_pcp = pol_scalar_product(d_X[a], new_rhat);
    auto noise = curand_uniform(&d_state[i])*0;
    auto more_along_pcp = fabs(new_pcp) > fabs(old_pcp)*(1.f - noise);
    if (not_initialized or more_along_pcp) {
        link->a = a;
        link->b = b;
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Po_cell, n_cells, Grid_solver> bolls;
    uniform_sphere(0.733333, bolls);
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].y /= 5;
        bolls.h_X[i].theta = acos(2.*rand()/(RAND_MAX + 1.) - 1.);
        bolls.h_X[i].phi = 2.*M_PI*rand()/(RAND_MAX + 1.);
    }
    bolls.copy_to_device();
    Links<static_cast<int>(n_cells*prots_per_cell)> protrusions;
    auto intercalation = std::bind(
        link_forces<static_cast<int>(n_cells*prots_per_cell), Po_cell, protrusion_force>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    // Simulate elongation
    Vtk_output output("aggregate");
    Grid<n_cells> grid;
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        protrusions.copy_to_host();

        grid.build(bolls, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(grid.d_grid,
            bolls.d_X, protrusions.d_link, protrusions.d_state);
        bolls.take_step<lb_force>(dt, intercalation);

        output.write_positions(bolls);
        output.write_links(protrusions);
        output.write_polarity(bolls);
    }

    return 0;
}
