// Relax bent epithelium
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 91;
const auto n_time_steps = 500;
const auto dt = 0.1;


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

    dF += rigidity_force(Xi, r, dist) * 0.5;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, n_cells, Tile_solver> bolls;
    regular_hexagon(0.75, bolls);
    auto radius = 1.6;
    for (auto i = 0; i < n_cells; i++) {
        // Rotate by π/6 to reduce negative curvature from tips
        auto x = bolls.h_X[i].x;
        auto y = bolls.h_X[i].y;
        bolls.h_X[i].x = cos(M_PI / 6) * x - sin(M_PI / 6) * y;
        bolls.h_X[i].y = sin(M_PI / 6) * x + cos(M_PI / 6) * y;

        // Wrap around cylinder
        auto phi = bolls.h_X[i].x / radius;
        if (phi == 0) phi = 0.01;
        bolls.h_X[i].x = radius * sin(phi);
        bolls.h_X[i].z = radius * cos(phi);
        bolls.h_X[i].theta = phi;
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
