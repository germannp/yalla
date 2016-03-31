// Integrate N-body problem with springs between all bodies.
#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float L_0 = 0.5;  // Relaxed spring length
const float DELTA_T = 0.001;
const uint N_CELLS = 800;
const uint N_TIME_STEPS = 100;

__device__ __managed__ Solution<float3, N_CELLS, N2nSolver> X;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r;
    r.x = Xi.x - Xj.x;
    r.y = Xi.y - Xj.y;
    r.z = Xi.z - Xj.z;
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF.x = r.x*(L_0 - dist)/dist;
    dF.y = r.y*(L_0 - dist)/dist;
    dF.z = r.z*(L_0 - dist)/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_spring = spring;


int main(int argc, const char* argv[]) {
    // Prepare initial state
    uniform_sphere(L_0, X);

    // Integrate positions
    VtkOutput output("springs");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, p_spring);
    }
}
