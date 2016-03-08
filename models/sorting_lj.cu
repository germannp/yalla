// Simulating cell sorting with Leonard-Jones potential.
#include <assert.h>
#include <cmath>
#include <curand_kernel.h>

#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MIN = 0.5;
const float MIN_DIST = 0.35;
const int N_CELLS = 200;
const int N_TIME_STEPS = 50000;
const int SKIP_STEPS = 100;
const float DELTA_T = 0.0001;

__device__ __managed__ Solution<float3, N_CELLS, N2nSolver> X;
__device__ __managed__ curandState rand_states[N_CELLS];


__device__ float3 lj_sorting(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    int strength = (1 + 2*(j < N_CELLS/2))*(1 + 2*(i < N_CELLS/2));
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dist = fmaxf(dist, MIN_DIST);
    float r_rel = R_MIN/dist;
    float F = powf(r_rel, 13);
    F -= powf(r_rel, 7);
    F += curand_normal(&rand_states[i])*10/sqrtf(N_CELLS);
    dF.x = strength*r.x*F/dist;
    dF.y = strength*r.y*F/dist;
    dF.z = strength*r.z*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_sorting = lj_sorting;


__global__ void setup_rand_states() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) curand_init(1337, i, 0, &rand_states[i]);
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, R_MIN, X);
    int cell_type[N_CELLS];
    for (int i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }
    setup_rand_states<<<(N_CELLS + 32 - 1)/32, 32>>>();
    cudaDeviceSynchronize();

    // Integrate cell positions
    VtkOutput output("sorting-lj", N_TIME_STEPS, SKIP_STEPS);
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        output.write_type(N_CELLS, cell_type);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, p_sorting);
    }
}
