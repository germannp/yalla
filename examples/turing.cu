// Simulate Meinhard equations within an epithelium
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


__device__ Epi_cell epithelium_w_turing(
    Epi_cell Xi, Epi_cell r, float dist, int i, int j)
{
    Epi_cell dF{0};

    // Meinhard equations
    if (i == j) {
        dF.u = lambda *
               ((f_u * Xi.u * Xi.u) / (1 + f_v * Xi.v) - m_u * Xi.u + s_u);
        dF.v = lambda * (g_u * Xi.u * Xi.u - m_v * Xi.v);

        return dF;
    }

    // Diffusion & mechanics
    if (dist > r_max) return dF;

    dF.u = -D_u * r.u;
    dF.v = -D_v * r.v;

    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force(Xi, r, dist) * 3;
    return dF;
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
    Vtk_output output{"turing"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<epithelium_w_turing>(dt);
        if (time_step % skip_steps == 0) {
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_field(cells, "u", &Epi_cell::u);
            output.write_field(cells, "v", &Epi_cell::v);
        }
    }

    return 0;
}
