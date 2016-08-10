// Simulate gradient formation
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_CELLS = 100;
const auto N_TIME_STEPS = 200;
const auto DELTA_T = 0.005;


__device__ float4 cubic_w_diffusion(float4 Xi, float4 Xj, int i, int j) {
    float4 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto n = 2;
    auto D = 10;
    auto strength = 100;
    auto F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
        + strength*powf(R_MAX - dist, n);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = i == 0 ? 0 : -r.w*D;
    return dF;
}

__device__ auto d_cubic_w_diffusion = &cubic_w_diffusion;
auto h_cubic_w_diffusion = get_device_object(d_cubic_w_diffusion, 0);


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float4, N_CELLS, N2nSolver> bolls;
    uniform_circle(0.733333, bolls);
    for (auto i = 0; i < N_CELLS; i++) {
        bolls.h_X[i].w = i == 0 ? 1 : 0;
    }
    bolls.memcpyHostToDevice();

    // Integrate cell positions
    VtkOutput output("gradient");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T, h_cubic_w_diffusion);
        output.write_positions(bolls);
        output.write_field(bolls);
    }

    return 0;
}
