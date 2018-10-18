// Simulate Meinhard equations within an epithelium
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto n_cells = 500;
const auto n_time_steps = 10000;
const auto skip_steps = 100;

const auto lambda = 1.;
const auto D_v = 4.0;
const auto f_v = 1.0;
const auto f_u = 80.0;
const auto g_u = 40.0;
const auto m_u = 0.25;
const auto m_v = 0.5;
const auto s_u = 0.05;
const auto D_u = 0.1;

const auto dt = 0.05 * r_min * r_min / D_v;

MAKE_PT(Epi_cell, theta, phi, u, v);


__device__ Epi_cell mechanics(Epi_cell Xi, Epi_cell r, float dist, int i, int j)
{
    Epi_cell dF{0};

    if (i == j) { return dF; }

    if (dist > r_max) return dF;

    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force(Xi, r, dist) * 3;
    return dF;
}


__device__ Epi_cell turing(Epi_cell Xi, Epi_cell r, float dist, int i, int j)
{
    Epi_cell dF{0};

    // Meinhard equations
    if (i == j) {
        dF.u = lambda *
               ((f_u * Xi.u * Xi.u) / (1 + f_v * Xi.v) - m_u * Xi.u + s_u);
        dF.v = lambda * (g_u * Xi.u * Xi.u - m_v * Xi.v);

        return dF;
    }

    // Diffusion
    if (dist > r_max) return dF;

    dF.u = -D_u * r.u;
    dF.v = -D_v * r.v;
    return dF;
}


__device__ Epi_cell additive_noise(Epi_cell Xi, float dt, curandState state)
{
    Xi.u *= 1 + 0.01 * curand_normal(&state) * sqrt(dt);
    Xi.v *= 1 + 0.01 * curand_normal(&state) * sqrt(dt);
    return Xi;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Epi_cell, Grid_solver> cells{n_cells};
    for (int i = 0; i < n_cells; i++) {
        cells.h_X[i].theta = M_PI / 2;
        cells.h_X[i].u = rand() / (RAND_MAX + 1.) / 5 - 0.1;
        cells.h_X[i].v = rand() / (RAND_MAX + 1.) / 5 - 0.1;
    }
    random_disk(0.5, cells);

    // Integrate positions
    Vtk_output output{"adaptive-turing"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<mechanics>(dt);
        // Beware, take_r12_step (or take_step) updates old velocities.
        // If turing leaves x, y, and z alone, this is introducing friction on
        // the background. Therefore splitting pairwise interactions should be
        // avoided!

        // If this cannot be avoided, do not update the old velocities in the
        // second step and use friction_on_background.

        // There is one more problem ...
        cells.take_rk12_step<turing, no_noise, friction_on_background>(dt);
        // cells.take_rk12_step<turing, additive_noise>(dt);
        if (time_step % skip_steps == 0) {
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_field(cells, "u", &Epi_cell::u);
            output.write_field(cells, "v", &Epi_cell::v);
        }
    }

    return 0;
}
