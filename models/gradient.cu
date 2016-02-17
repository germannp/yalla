// Simulating a layer.
#include <assert.h>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 100;
const int N_TIME_STEPS = 200;
const float DELTA_T = 0.005;

__device__ __managed__ Solution<float4, N_CELLS, N2nSolver> X;


__device__ float4 cubic_w_diffusion(float4 Xi, float4 Xj, int i, int j) {
    float4 dF = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z, Xi.w - Xj.w};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (i != j) {
        int n = 2;
        float strength = 100;
        float F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
            + strength*powf(R_MAX - dist, n);
        float D = dist < 1 ? 10 : 0;
        dF.x = r.x*F/dist;
        dF.y = r.y*F/dist;
        dF.z = r.z*F/dist;
        dF.w = i == 0 ? 0 : -r.w*D;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float4> local = cubic_w_diffusion;


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_circle(N_CELLS, 0.733333, X);
    for (int i = 0; i < N_CELLS; i++) {
        X[i].w = i == 0 ? 1 : 0;
    }

    // Integrate cell positions
    VtkOutput output("gradient");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        output.write_field(N_CELLS, "w", X);

        if (time_step < N_TIME_STEPS) {
            X.step(DELTA_T, N_CELLS, local);
        }
    }

    return 0;
}
