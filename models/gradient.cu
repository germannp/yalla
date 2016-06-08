// Simulate gradient formation
#include <assert.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_CELLS = 100;
const auto N_TIME_STEPS = 200;
const auto DELTA_T = 0.005;

__device__ __managed__ Solution<float4, N_CELLS, N2nSolver> X;


__device__ float4 cubic_w_diffusion(float4 Xi, float4 Xj, int i, int j) {
    float4 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if  (dist > R_MAX) return dF;

    auto n = 2;
    auto D = 10;
    auto strength = 100;
    auto F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
        + strength*powf(R_MAX - dist, n);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = i == 0 ? 0 : -r.w*D;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ auto d_local_interactions = cubic_w_diffusion;


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_circle(0.733333, X);
    for (auto i = 0; i < N_CELLS; i++) {
        X[i].w = i == 0 ? 1 : 0;
    }

    // Integrate cell positions
    VtkOutput output("gradient");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_field(X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, d_local_interactions);
    }
}
