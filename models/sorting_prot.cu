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

__device__ __managed__ Solution<float3, N_CELLS, LatticeSolver> X;
__device__ __managed__ Protrusions<N_LINKS> prots;


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

__device__ __managed__ auto d_cubic = cubic;


__global__ void update_links() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_LINKS) return;

    auto r = curand_uniform(&prots.rand_states[i]);
    auto j = prots.links[i][0];
    auto k = prots.links[i][1];
    if ((j < N_CELLS/2) and (k < N_CELLS/2)) {
        if (r > 0.05) return;
    } else if ((j > N_CELLS/2) and (k > N_CELLS/2)) {
        if (r > 0.25) return;
    } else {
        if (r > 0.75) return;
    }

    auto new_j = static_cast<int>(curand_uniform(&prots.rand_states[i])*N_CELLS);
    auto new_k = static_cast<int>(curand_uniform(&prots.rand_states[i])*N_CELLS);
    if (new_j == new_k) return;

    auto dx = X[new_j] - X[new_k];
    auto dist = sqrtf(dx.x*dx.x + dx.y*dx.y + dx.z*dx.z);
    if (dist > 2) return;

    prots.links[i][0] = new_j;
    prots.links[i][1] = new_k;
}

void prots_forces(const float3* __restrict__ X, float3* dX) {
    intercalate<<<(N_LINKS + 32 - 1)/32, 32>>>(X, dX, prots);
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, X);
    int cell_type[N_CELLS];
    for (auto i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }
    init_protrusions(prots);

    // Integrate cell positions
    VtkOutput output("sorting");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_protrusions(prots);
        output.write_type(cell_type);
        if (time_step == N_TIME_STEPS) return 0;

        // X.step(DELTA_T, d_cubic);
        X.step(DELTA_T, d_cubic, prots_forces);
        update_links<<<(N_LINKS + 32 - 1)/32, 32>>>();
        cudaDeviceSynchronize();
    }
}
