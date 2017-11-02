// Simulate gradient formation
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto D = 10;
const auto n_cells = 61;
const auto n_time_steps = 200;
const auto dt = 0.005;


__device__ float4 diffusion(float4 Xi, float4 r, float dist, int i, int j)
{
    float4 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    dF.w = i == 11 ? 0 : -r.w * D;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float4, n_cells, Tile_solver> bolls;
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = i == 11 ? 1 : 0;
    }
    regular_hexagon(0.75, bolls);

    // Integrate cell positions
    Vtk_output output("gradient");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<diffusion>(dt);
        output.write_positions(bolls);
        output.write_field(bolls);
    }

    return 0;
}
