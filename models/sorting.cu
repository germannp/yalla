// Simulating cell sorting with limited interactions.
#include <assert.h>
#include <cmath>
#include <sys/stat.h>
#include <thrust/sort.h>

#include "../lib/vtk.cuh"
#include "../lib/lattice.cuh"


const float R_MAX = 1;
const float R_MIN = 0.5;
const int N_CELLS = 1000;
const int N_TIME_STEPS = 300;
const float DELTA_T = 0.05;

__device__ __managed__ float3 X[N_CELLS];


__device__ float3 cell_cell_interaction(float3 Xi, float3 Xj) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (dist > 1e-8) {
        float F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
        dF.x += r.x*F/dist;
        dF.y += r.y*F/dist;
        dF.z += r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    int cell_type[N_CELLS];
    float r_sphere = pow(N_CELLS/0.75, 1./3)*R_MIN/2; // Sphere packing
    for (int i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
        float r = r_sphere*pow(rand()/(RAND_MAX + 1.), 1./3);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }

    // Integrate cell positions
    mkdir("output", 755);
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        char file_name[22];
        sprintf(file_name, "output/sorting_%03i.vtk", time_step);
        write_positions(file_name, N_CELLS, X);
        write_scalars(file_name, N_CELLS, "cell_type", cell_type);

        if (time_step < N_TIME_STEPS) {
            euler_step(DELTA_T, N_CELLS, X);
        }
    }

    return 0;
}
