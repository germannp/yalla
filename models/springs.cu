// Integrate N-body problem with springs between all bodies
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto L_0 = 0.5f;  // Relaxed spring length
const auto N_CELLS = 800u;
const auto N_TIME_STEPS = 100u;
const auto DELTA_T = 0.001f;

__device__ __managed__ Solution<float3, N_CELLS, N2nSolver> X;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF = r*(L_0 - dist)/dist;
    return dF;
}

__device__ __managed__ auto d_spring = spring;  // Copy to device


int main(int argc, const char* argv[]) {
    // Prepare initial state
    uniform_sphere(L_0, X);

    // Integrate positions
    VtkOutput output("springs");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, d_spring);
    }
}
