// Integrate N-body problem with springs between all bodies
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto L_0 = 0.5f;  // Relaxed spring length
const auto N_CELLS = 800u;
const auto N_TIME_STEPS = 100u;
const auto DELTA_T = 0.001f;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF = r*(L_0 - dist)/dist;
    return dF;
}

__device__ auto d_spring = &spring;
auto h_spring = get_device_object(d_spring, 0);


int main(int argc, const char* argv[]) {
    // Prepare initial state
    Solution<float3, N_CELLS, N2nSolver> bolls;
    uniform_sphere(L_0, bolls);

    // Integrate positions
    VtkOutput output("springs");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T, h_spring);  // Writing starts during calculation, use thread
        output.write_positions(bolls);  // for full concurency.
    }

    return 0;
}
