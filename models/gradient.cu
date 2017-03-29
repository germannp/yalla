// Simulate gradient formation
#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto D = 10;
const auto strength = 100;
const auto n_cells = 100;
const auto n_time_steps = 200;
const auto dt = 0.005;


__device__ float4 clipped_cubic_w_gradient(float4 Xi, float4 Xj, int i, int j) {
    float4 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = strength*(2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2));
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = i == 0 ? 0 : -r.w*D;
    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float4, n_cells, N2n_solver> bolls;
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = i == 0 ? 1 : 0;
    }
    uniform_circle(0.733333, bolls);

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
