// Simulate cell sorting by protrusions
#include <functional>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/inits.cuh"
#include "../lib/links.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 200u;
const auto n_protrusions = n_cells*5;
const auto n_time_steps = 300u;
const auto dt = 0.05;


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + (r_max - dist)*(r_max - dist);
    dF = r*F/dist;
    return dF;
}


__global__ void update_protrusions(const float3* __restrict__ d_X, Link* d_link,
        curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_protrusions) return;

    auto r = curand_uniform(&d_state[i]);
    auto j = d_link[i].a;
    auto k = d_link[i].b;
    if ((j < n_cells/2) and (k < n_cells/2)) {
        if (r > 0.05) return;
    } else if ((j > n_cells/2) and (k > n_cells/2)) {
        if (r > 0.25) return;
    } else {
        if (r > 0.75) return;
    }

    auto new_j = min(static_cast<int>(curand_uniform(&d_state[i])*n_cells), n_cells - 1);
    auto new_k = min(static_cast<int>(curand_uniform(&d_state[i])*n_cells), n_cells - 1);
    if (new_j == new_k) return;

    auto dx = d_X[new_j] - d_X[new_k];
    auto dist = norm3df(dx.x, dx.y, dx.z);
    if (dist > 2) return;

    d_link[i].a = new_j;
    d_link[i].b = new_k;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, n_cells, Lattice_solver> bolls;
    uniform_sphere(r_min, bolls);
    Links<n_protrusions> protrusions;
    auto prot_forces = std::bind(linear_force<n_protrusions>, protrusions,
        std::placeholders::_1, std::placeholders::_2);
    Property<n_cells> type;
    for (auto i = 0; i < n_cells; i++) {
        type.h_prop[i] = (i < n_cells/2) ? 0 : 1;
    }

    // Integrate cell positions
    Vtk_output output("sorting");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        protrusions.copy_to_host();
        update_protrusions<<<(n_protrusions + 32 - 1)/32, 32>>>(bolls.d_X, protrusions.d_link,
            protrusions.d_state);
        bolls.take_step<clipped_cubic>(dt, prot_forces);
        output.write_positions(bolls);
        output.write_links(protrusions);
        output.write_property(type);
    }

    return 0;
}
