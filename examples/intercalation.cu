// Simulate intercalating cells
#include <functional>
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 500u;
const auto prots_per_cell = 1;
const auto n_time_steps = 250u;
const auto dt = 0.2f;


__device__ float3 clipped_cubic(float3 Xi, float3 r, float dist, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + (r_max - dist)*(r_max - dist);
    dF = r*F/dist;
    return dF;
}


__global__ void update_protrusions(const float3* __restrict__ d_X, Link* d_link,
        curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5)/prots_per_cell);
    auto k = min(static_cast<int>(curand_uniform(&d_state[i])*n_cells), n_cells - 1);
    if (j == k) return;

    auto r = d_X[j] - d_X[k];
    auto dist = norm3df(r.x, r.y, r.z);
    if ((fabs(r.x/dist) < 0.2) and (dist < 2)) {
        d_link[i].a = j;
        d_link[i].b = k;
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, n_cells, Lattice_solver> bolls;
    uniform_sphere(r_min, bolls);
    Links<n_cells*prots_per_cell> protrusions;
    auto intercalation = std::bind(linear_force<n_cells*prots_per_cell>, protrusions,
        std::placeholders::_1, std::placeholders::_2);

    // Integrate cell positions
    Vtk_output output("intercalation");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        protrusions.copy_to_host();
        update_protrusions<<<(n_cells*prots_per_cell + 32 - 1)/32, 32>>>(bolls.d_X,
            protrusions.d_link, protrusions.d_state);
        bolls.take_step<clipped_cubic>(dt, intercalation);
        output.write_positions(bolls);
        output.write_links(protrusions);
    }

    return 0;
}
