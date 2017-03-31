// Simulate cell sorting by forces strength
#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "../include/property.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 100u;
const auto n_time_steps = 300u;
const auto dt = 0.05;


__device__ float3 differential_adhesion(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto strength = (1 + 2*(j < n_cells/2))*(1 + 2*(i < n_cells/2));
    auto F = 2*(r_min - dist)*(r_max - dist) + (r_max - dist)*(r_max - dist);
    dF = strength*r*F/dist;
    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, n_cells, Lattice_solver> bolls;
    uniform_sphere(r_min, bolls);
    Property<n_cells> type;
    for (auto i = 0; i < n_cells; i++) {
        type.h_prop[i] = (i < n_cells/2) ? 0 : 1;
    }

    // Integrate cell positions
    Vtk_output output("sorting");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<differential_adhesion>(dt);
        output.write_positions(bolls);
        output.write_property(type);
    }

    return 0;
}
