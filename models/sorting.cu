// Simulate cell sorting
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 100u;
const auto N_TIME_STEPS = 300u;
const auto DELTA_T = 0.05;

Solution<float3, N_CELLS, LatticeSolver> bolls;
Property<N_CELLS> type;


// Sorting by forces strength
__device__ float3 cubic_sorting(float3 Xi, float3 Xj, int i, int j) {
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

__device__ auto d_cubic_sorting = &cubic_sorting;
auto h_cubic_sorting = get_device_object(d_cubic_sorting, 0);


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, bolls);
    for (auto i = 0; i < N_CELLS; i++) {
        type.h_prop[i] = (i < N_CELLS/2) ? 0 : 1;
    }

    // Integrate cell positions
    VtkOutput output("sorting");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T, h_cubic_sorting);
        output.write_positions(bolls);
        output.write_property(type);
    }

    return 0;
}
