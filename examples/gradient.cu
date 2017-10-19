// Simulate gradient formation
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto D = 10;
const auto n_cells = 100;
const auto n_time_steps = 200;
const auto dt = 0.005;


__device__ float4 clipped_cubic_w_gradient(
    float4 Xi, float4 r, float dist, int i, int j)
{
    float4 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;
    dF.w = i == 0 ? 0 : -r.w * D;
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float4, n_cells, Tile_solver> bolls;
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = i == 0 ? 1 : 0;
    }
    random_disk(0.5, bolls);

    // Integrate cell positions
    Vtk_output output("gradient");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<clipped_cubic_w_gradient>(dt);
        output.write_positions(bolls);
        output.write_field(bolls);
    }

    return 0;
}
