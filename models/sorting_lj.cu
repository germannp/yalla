// Simulate cell sorting with Leonard-Jones potential
#include <curand_kernel.h>
#include <assert.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto R_MIN = 0.5;
const auto MIN_DIST = 0.35;
const auto N_CELLS = 200u;
const auto N_TIME_STEPS = 50000u;
const auto SKIP_STEPS = 100u;
const auto DELTA_T = 0.0001;

__device__ __managed__ Solution<float3, N_CELLS, N2nSolver> X;
__device__ curandState rand_states[N_CELLS];


__device__ float3 lj_sorting(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto strength = (1 + 2*(j < N_CELLS/2))*(1 + 2*(i < N_CELLS/2));
    auto r = Xi - Xj;
    auto dist = fmaxf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), MIN_DIST);  // Smoothen core
    auto r_rel = R_MIN/dist;
    auto F = powf(r_rel, 13);
    F -= powf(r_rel, 7);
    F += curand_normal(&rand_states[i])*10/sqrtf(N_CELLS);
    dF = strength*r*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ auto d_sorting = lj_sorting;


__global__ void setup_rand_states() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) curand_init(1337, i, 0, &rand_states[i]);
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, X);
    int cell_type[N_CELLS];
    for (auto i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }
    setup_rand_states<<<(N_CELLS + 32 - 1)/32, 32>>>();
    cudaDeviceSynchronize();

    // Integrate cell positions
    VtkOutput output("sorting-lj", N_TIME_STEPS, SKIP_STEPS);
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_type(cell_type);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, d_sorting);
    }
}
