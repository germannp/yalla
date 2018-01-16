// Simulate cell sorting by forces strength
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 100u;
const auto n_time_steps = 300u;
const auto dt = 0.05;


__device__ float3 differential_adhesion(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto strength = (1 + 2 * (j < n_cells / 2)) * (1 + 2 * (i < n_cells / 2));
    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF = strength * r * F / dist;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float3, Grid_solver> cells{n_cells};
    random_sphere(r_min, cells);
    Property<> type{n_cells};
    for (auto i = 0; i < n_cells; i++) {
        type.h_prop[i] = (i < n_cells / 2) ? 0 : 1;
    }

    // Integrate cell positions
    Vtk_output output{"sorting"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<differential_adhesion>(dt);
        output.write_positions(cells);
        output.write_property(type);
    }

    return 0;
}
