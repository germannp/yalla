// Simulate intercalating cells
#include <functional>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/protrusions.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 500u;
const auto N_LINKS = 250u;
const auto N_TIME_STEPS = 1000u;
const auto DELTA_T = 0.2f;


__device__ float3 pairwise_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF = r*F/dist;
    return dF;
}

#include "../lib/solvers.cuh"


__global__ void update_links(const float3* __restrict__ d_X, Link* d_link,
        curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_LINKS) return;

    auto j = min(static_cast<int>(curand_uniform(&d_state[i])*N_CELLS), N_CELLS - 1);
    auto k = min(static_cast<int>(curand_uniform(&d_state[i])*N_CELLS), N_CELLS - 1);
    auto r = d_X[j] - d_X[k];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if ((fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
        d_link[i].a = j;
        d_link[i].b = k;
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, N_CELLS, LatticeSolver> bolls;
    uniform_sphere(R_MIN, bolls);
    Protrusions<N_LINKS> links;
    auto intercalation = std::bind(link_forces<N_LINKS>, links,
        std::placeholders::_1, std::placeholders::_2);
    int i = 0;
    while (i < N_LINKS) {
        auto j = static_cast<int>(rand()/(RAND_MAX + 1.)*N_CELLS);
        auto k = static_cast<int>(rand()/(RAND_MAX + 1.)*N_CELLS);
        auto r = bolls.h_X[j] - bolls.h_X[k];
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if ((fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
            links.h_link[i].a = j;
            links.h_link[i].b = k;
            i++;
        }
    }
    links.memcpyHostToDevice();

    // Integrate cell positions
    VtkOutput output("intercalation");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        links.memcpyDeviceToHost();
        update_links<<<(N_LINKS + 32 - 1)/32, 32>>>(bolls.d_X, links.d_link, links.d_state);
        bolls.step(DELTA_T, intercalation);
        output.write_positions(bolls);
        output.write_protrusions(links);
    }

    return 0;
}
