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
__device__ Po_cell rigid_relu_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += rigidity_force(Xi, r, dist) * 0.2;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, n_cells, Grid_solver> bolls;
    relaxed_sphere(0.8, bolls);
    for (auto i = 0; i < n_cells; i++) {
        auto dist = sqrtf(bolls.h_X[i].x * bolls.h_X[i].x +
                          bolls.h_X[i].y * bolls.h_X[i].y +
                          bolls.h_X[i].z * bolls.h_X[i].z);
        bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x) +
                           rand() / (RAND_MAX + 1.) * 0.5;
        bolls.h_X[i].theta =
            acosf(bolls.h_X[i].z / dist) + rand() / (RAND_MAX + 1.) * 0.5;
    }
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("epithelium");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<rigid_relu_force>(dt);
        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
