// Simulate cell sorting by protrusions
#include <functional>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/protrusions.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 200u;
const auto N_LINKS = N_CELLS*5;
const auto N_TIME_STEPS = 300u;
const auto DELTA_T = 0.05;

Solution<float3, N_CELLS, LatticeSolver> bolls;
Protrusions<N_LINKS> links;


__device__ float3 cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF = r*F/dist;
    return dF;
}

__device__ auto d_cubic = &cubic;
auto h_cubic = get_device_object(d_cubic, 0);


__global__ void update_links(const float3* __restrict__ d_X, Link* d_cell_id,
        curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_LINKS) return;

    auto r = curand_uniform(&d_state[i]);
    auto j = d_cell_id[i].a;
    auto k = d_cell_id[i].b;
    if ((j < N_CELLS/2) and (k < N_CELLS/2)) {
        if (r > 0.05) return;
    } else if ((j > N_CELLS/2) and (k > N_CELLS/2)) {
        if (r > 0.25) return;
    } else {
        if (r > 0.75) return;
    }

    auto new_j = static_cast<int>(curand_uniform(&d_state[i])*N_CELLS);
    auto new_k = static_cast<int>(curand_uniform(&d_state[i])*N_CELLS);
    if (new_j == new_k) return;

    auto dx = d_X[new_j] - d_X[new_k];
    auto dist = sqrtf(dx.x*dx.x + dx.y*dx.y + dx.z*dx.z);
    if (dist > 2) return;

    d_cell_id[i].a = new_j;
    d_cell_id[i].b = new_k;
}

void links_forces(const float3* __restrict__ d_X, float3* d_dX) {
    link_force<<<(N_LINKS + 32 - 1)/32, 32>>>(d_X, d_dX, links.d_cell_id, N_LINKS);
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, bolls);
    int cell_type[N_CELLS];
    for (auto i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }

    // Integrate cell positions
    VtkOutput output("sorting");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        links.memcpyDeviceToHost();
        bolls.step(DELTA_T, h_cubic, links_forces);
        update_links<<<(N_LINKS + 32 - 1)/32, 32>>>(bolls.d_X, links.d_cell_id, links.d_state);
        output.write_positions(bolls);
        output.write_protrusions(links);
        output.write_type(cell_type);
    }

    return 0;
}