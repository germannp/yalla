// Simulate a apical constriction in an epithelial layer
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 225;
const auto n_time_steps = 4000;
const auto dt = 0.1;
const auto preferential_angle_deviation = 20.f * M_PI/180.f; // in radiants

__device__ Po_cell constriction_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.8 - dist, 0) * 2 - fmaxf(dist - 0.8, 0)*2;
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    float cos_alfa = abs(r.x) / dist;

    dF += apical_constriction_force(Xi, r, dist,
        M_PI/2.f - preferential_angle_deviation) * 0.6 ;

    return dF;
}

int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Grid_solver> cells{n_cells};
    regular_rectangle(0.8, 15, cells);

    for (auto i = 0; i < n_cells; i++) {
        cells.h_X[i].theta = acosf(1.0);
        cells.h_X[i].phi = atan2(1.0, 1.0);
    }
    cells.copy_to_device();

    // Integrate cell positions
    Vtk_output output{"apical_constriction", "output/", false};

    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if(time_step%40 == 0)
            cells.copy_to_host();

        cells.take_step<constriction_force, friction_on_background>(dt);

        if(time_step%40 == 0){
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_field(cells, "z", &Po_cell::z);  // For visualization
        }
    }

    return 0;
}
