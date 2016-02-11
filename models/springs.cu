// Integrate N-body problem with springs between all bodies.
#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float L_0 = 0.5; // Relaxed spring length
const float DELTA_T = 0.001;
const uint N_CELLS = 800;
const uint N_TIME_STEPS = 100;

__device__ __managed__ float3 X[N_CELLS];
__device__ __managed__ N2nSolver<float3, N_CELLS> solver;


__device__ float3 neighbourhood_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 r;
    float3 dF = {0.0f, 0.0f, 0.0f};
    r.x = Xi.x - Xj.x;
    r.y = Xi.y - Xj.y;
    r.z = Xi.z - Xj.z;
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (i != j) {
        dF.x = r.x*(L_0 - dist)/dist;
        dF.y = r.y*(L_0 - dist)/dist;
        dF.z = r.z*(L_0 - dist)/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}


void global_interactions(const float3* __restrict__ X, float3* dX) {}


int main(int argc, const char* argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, L_0, X);

    // Integrate positions
    VtkOutput output("springs");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);

        if (time_step < N_TIME_STEPS) {
            solver.step(DELTA_T, N_CELLS, X);
        }
    }

    return 0;
}
