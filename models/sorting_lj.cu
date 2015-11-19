// Simulating cell sorting with Leonard-Jones potential.
#include <assert.h>
#include <cmath>
#include <sys/stat.h>

#include "../lib/sphere.cuh"
#include "../lib/vtk.cuh"
#include "../lib/n2n.cuh"


const float R_MIN = 0.5;
const float MIN_DIST = 0.35;
const int N_CELLS = 200;
const int N_TIME_STEPS = 50000;
const int SKIP_STEPS = 100;
const float DELTA_T = 0.0001;

__device__ __managed__ float3 X[N_CELLS];


__device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j) {
    int strength = (1 + 2*(j < N_CELLS/2))*(1 + 2*(i < N_CELLS/2));
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fmaxf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), MIN_DIST);
    if (dist > 1e-7) {
        float r_rel = R_MIN/dist;
        float F = powf(r_rel, 12);
        F -= 2*powf(r_rel, 6);
        dF.x = strength*r.x*F/dist;
        dF.y = strength*r.y*F/dist;
        dF.z = strength*r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, R_MIN, X);
    int cell_type[N_CELLS];
    for (int i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }

    // Integrate cell positions
    mkdir("output", 755);
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        if (time_step % SKIP_STEPS == 0) {
            char file_name[25];
            sprintf(file_name, "output/sorting-lj_%03i.vtk", time_step/SKIP_STEPS);
            write_positions(file_name, N_CELLS, X);
            write_scalars(file_name, N_CELLS, "cell_type", cell_type);
        }

        if (time_step < N_TIME_STEPS) {
            euler_step(DELTA_T, N_CELLS, X);
        }
    }

    return 0;
}