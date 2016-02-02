// Integrate N-body problem with springs between all bodies.
#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"
#include "../lib/n2n.cuh"


const float L_0 = 0.5; // Relaxed spring length
const float DELTA_T = 0.001;
const uint N_CELLS = 800;
const uint N_TIME_STEPS = 100;

__device__ __managed__ float3 X[N_CELLS], dX[N_CELLS], X1[N_CELLS], dX1[N_CELLS];


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


void global_interactions(const __restrict__ float3* X, float3* dX) {}


int main(int argc, const char* argv[]) {
    assert(N_CELLS % TILE_SIZE == 0);

    // Prepare initial state
    uniform_sphere(N_CELLS, L_0, X);

    // Integrate positions
    VtkOutput output("springs");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);

        if (time_step < N_TIME_STEPS) {
            heun_step(DELTA_T, N_CELLS, X, dX, X1, dX1);
        }
    }

    return 0;
}
