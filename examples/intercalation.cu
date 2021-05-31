// Simulate intercalating cells
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.f;
const auto r_min = 0.5f;
const auto n_cells = 500u;
const auto prots_per_cell = 1;
const auto n_time_steps = 250u;
const auto dt = 0.2f;


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
    if (i >= n_cells * prots_per_cell) return;

    auto r = d_X[d_link[i].a] - d_X[d_link[i].b];
    auto dist = norm3df(r.x, r.y, r.z);
    if ((dist < 1) or (dist > 2)) {
        d_link[i].a = 0;
        d_link[i].b = 0;
    }

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto k = min(
        static_cast<int>(curand_uniform(&d_state[i]) * n_cells), n_cells - 1);
    if (j == k) return;

    r = d_X[j] - d_X[k];
    dist = norm3df(r.x, r.y, r.z);
    if ((fabs(r.x / dist) < 0.2) and (1 < dist < 2)) {
        d_link[i].a = j;
        d_link[i].b = k;
    }
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float3, Grid_solver> cells{n_cells};
    random_sphere(r_min, cells);
    Links protrusions{n_cells * prots_per_cell};
    auto intercalation = [&protrusions](
                             const int n, const float3* __restrict__ d_X, float3* d_dX) {
        return link_forces(protrusions, d_X, d_dX);
    };

    // Integrate cell positions
    Vtk_output output{"intercalation"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        protrusions.copy_to_host();
        update_protrusions<<<(n_cells * prots_per_cell + 32 - 1) / 32, 32>>>(
            cells.d_X, protrusions.d_state, protrusions.d_link);
        cells.take_step<clipped_cubic>(dt, intercalation);
        output.write_positions(cells);
        output.write_links(protrusions);
    }

    return 0;
}
