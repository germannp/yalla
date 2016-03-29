// Simulating a layer.
#include <assert.h>
#include <cmath>
#include <iostream>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 1000;
const int N_TIME_STEPS = 200;
const float DELTA_T = 0.005;

__device__ __managed__ Solution<float3, N_CELLS, LatticeSolver> X;


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    int n = 2;
    float strength = 100;
    float F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
        + strength*powf(R_MAX - dist, n);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> potential = clipped_cubic;


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_circle(N_CELLS, 0.733333/1.5, X);
    for (int i = 0; i < N_CELLS; i++) {
        X[i].x = sin(X[i].y);
    }

    // Integrate cell positions
    VtkOutput output("layer");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, potential);
    }
}
