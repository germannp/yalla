// Simulate cell sorting by protrusions
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 200u;
const auto n_protrusions = n_cells * 5;
const auto n_time_steps = 300u;
const auto dt = 0.05;


__device__ float3 clipped_cubic(float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = 2 * (r_min - dist) * (r_max - dist) + powf(r_max - dist, 2);
    dF = r * F / dist;
    return dF;
}


__global__ void update_protrusions(
    const float3* __restrict__ d_X, curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_protrusions) return;

    auto r = d_X[d_link[i].a] - d_X[d_link[i].b];
    auto dist = norm3df(r.x, r.y, r.z);
    if ((dist < 1) or (dist > 2)) {
        d_link[i].a = 0;
        d_link[i].b = 0;
    }

    auto rnd = curand_uniform(&d_state[i]);
    auto j = d_link[i].a;
    auto k = d_link[i].b;
    if ((j < n_cells / 2) and (k < n_cells / 2)) {
        if (rnd > 0.05) return;
    } else if ((j > n_cells / 2) and (k > n_cells / 2)) {
        if (rnd > 0.25) return;
    } else {
        if (rnd > 0.125) return;
    }

    auto new_j = min(
        static_cast<int>(curand_uniform(&d_state[i]) * n_cells), n_cells - 1);
    auto new_k = min(
        static_cast<int>(curand_uniform(&d_state[i]) * n_cells), n_cells - 1);
    if (new_j == new_k) return;

    r = d_X[new_j] - d_X[new_k];
    dist = norm3df(r.x, r.y, r.z);
    if (1 < dist < 2) {
        d_link[i].a = new_j;
        d_link[i].b = new_k;
    }
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float3, Grid_solver> cells{n_cells};
    random_sphere(r_min, cells);
    Links protrusions{n_protrusions};
    auto prot_forces = [&protrusions](
                           const int n, const float3* __restrict__ d_X, float3* d_dX) {
        return link_forces(protrusions, d_X, d_dX);
    };
    Property<> type{n_cells};
    for (auto i = 0; i < n_cells; i++) {
        type.h_prop[i] = (i < n_cells / 2) ? 0 : 1;
    }

    // Integrate cell positions
    Vtk_output output{"sorting"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        protrusions.copy_to_host();
        update_protrusions<<<(n_protrusions + 32 - 1) / 32, 32>>>(
            cells.d_X, protrusions.d_state, protrusions.d_link);
        cells.take_step<clipped_cubic>(dt, prot_forces);
        output.write_positions(cells);
        output.write_links(protrusions);
        output.write_property(type);
    }

    return 0;
}
