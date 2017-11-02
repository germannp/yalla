// Simulate planer cell polarity aligned by gradient
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto n_cells = 200;
const auto n_time_steps = 300;
const auto dt = 0.025;


__device__ Po_cell pcp(Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    // U_PCP = - Σ(n_i . n_j)^2/2
    dF += pcp_force(Xi, Xi - r);
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, n_cells, Grid_solver> bolls;
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].theta = acos(2. * rand() / (RAND_MAX + 1.) - 1.);
        bolls.h_X[i].phi = 2. * M_PI * rand() / (RAND_MAX + 1.);
    }
    random_sphere(0.5, bolls);

    // Integrate cell positions
    Vtk_output output("pcp");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<pcp>(dt);
        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
