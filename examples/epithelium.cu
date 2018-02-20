// Simulate a mesenchyme-to-epithelium transition
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 250;
const auto n_time_steps = 100;
const auto dt = 0.05;


// ReLU forces plus k*(n_i . r_ij/r)^2/2 for all r_ij <= r_max
__device__ Po_cell layer_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force(Xi, r, dist) * 0.2;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Grid_solver> cells{n_cells};
    relaxed_sphere(0.8, cells);
    for (auto i = 0; i < n_cells; i++) {
        auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                          cells.h_X[i].y * cells.h_X[i].y +
                          cells.h_X[i].z * cells.h_X[i].z);
        cells.h_X[i].theta =
            acosf(cells.h_X[i].z / dist) + rand() / (RAND_MAX + 1.) * 0.5;
        cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x) +
                           rand() / (RAND_MAX + 1.) * 0.5;
    }
    cells.copy_to_device();

    // Integrate cell positions
    Vtk_output output{"epithelium"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<layer_force, friction_on_background>(dt);
        output.write_positions(cells);
        output.write_polarity(cells);
        output.write_field(cells, "z", &Po_cell::z);  // For visualization
    }

    return 0;
}
