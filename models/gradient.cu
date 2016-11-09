// Simulate gradient formation
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto n_cells = 100;
const auto n_time_steps = 200;
const auto dt = 0.005;


__device__ float4 pairwise_interaction(float4 Xi, float4 Xj, int i, int j) {
    float4 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > r_max) return dF;

    auto n = 2;
    auto D = 10;
    auto strength = 100;
    auto F = strength*n*(r_min - dist)*powf(r_max - dist, n - 1)
        + strength*powf(r_max - dist, n);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = i == 0 ? 0 : -r.w*D;
    return dF;
}

#include "../lib/solvers.cuh"


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float4, n_cells, N2n_solver> bolls;
    uniform_circle(0.733333, bolls);
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = i == 0 ? 1 : 0;
    }
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("gradient");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step(dt);
        output.write_positions(bolls);
        output.write_field(bolls);
    }

    return 0;
}
