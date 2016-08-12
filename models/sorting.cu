// Simulate cell sorting by forces strength
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 100u;
const auto N_TIME_STEPS = 300u;
const auto DELTA_T = 0.05;


__device__ float3 pairwise_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto strength = (1 + 2*(j < N_CELLS/2))*(1 + 2*(i < N_CELLS/2));
    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF = strength*r*F/dist;
    return dF;
}

#include "../lib/solvers.cuh"


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, N_CELLS, LatticeSolver> bolls;
    uniform_sphere(R_MIN, bolls);
    Property<N_CELLS> type;
    for (auto i = 0; i < N_CELLS; i++) {
        type.h_prop[i] = (i < N_CELLS/2) ? 0 : 1;
    }

    // Integrate cell positions
    VtkOutput output("sorting");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T);
        output.write_positions(bolls);
        output.write_property(type);
    }

    return 0;
}
