// Simulate intercalating cells
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/protrusions.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 500u;
const auto N_LINKS = 250u;
const auto N_TIME_STEPS = 1000u;
const auto DELTA_T = 0.2f;

__device__ __managed__ Solution<float3, N_CELLS, LatticeSolver> X;
__device__ __managed__ Protrusions<N_LINKS> prots;


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF = r*F/dist;
    return dF;
}

__device__ __managed__ auto d_potential = clipped_cubic;


__global__ void update_links() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_LINKS) return;

    auto j = static_cast<int>(curand_uniform(&prots.rand_states[i])*N_CELLS);
    auto k = static_cast<int>(curand_uniform(&prots.rand_states[i])*N_CELLS);
    auto r = X[j] - X[k];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if ((fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
        prots.links[i][0] = j;
        prots.links[i][1] = k;
    }
}

void intercalation(const float3* __restrict__ X, float3* dX) {
    intercalate<<<(N_LINKS + 32 - 1)/32, 32>>>(X, dX, prots);
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, X);
    init_protrusions(prots);
    int i = 0;
    while (i < N_LINKS) {
        auto j = static_cast<int>(rand()/(RAND_MAX + 1.)*N_CELLS);
        auto k = static_cast<int>(rand()/(RAND_MAX + 1.)*N_CELLS);
        auto r = X[j] - X[k];
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if ((fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
            prots.links[i][0] = j;
            prots.links[i][1] = k;
            i++;
        }
    }

    // Integrate cell positions
    VtkOutput output("intercalation");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_protrusions(prots);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, d_potential, intercalation);
        update_links<<<(N_LINKS + 32 - 1)/32, 32>>>();
        cudaDeviceSynchronize();
    }
}
