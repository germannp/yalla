// Simulate mono-polar, amoeboid migration
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 91;
const auto n_time_steps = 300;
const auto dt = 0.005;


__device__ Po_cell relu_w_migration(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += migration_force(Xi, r, dist);
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, n_cells, Tile_solver> bolls;
    regular_hexagon(0.75, bolls);
    auto last = n_cells - 1;
    auto dist = sqrtf(bolls.h_X[last].x * bolls.h_X[last].x +
                      bolls.h_X[last].y * bolls.h_X[last].y +
                      bolls.h_X[last].z * bolls.h_X[last].z);
    bolls.h_X[last].theta = acosf(-bolls.h_X[last].z / dist);
    bolls.h_X[last].phi = atan2(-bolls.h_X[last].y, -bolls.h_X[last].x);
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("amoeboid_migration");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<relu_w_migration>(dt);
        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}