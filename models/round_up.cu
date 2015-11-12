// Simulating cell sorting with limited interactions.
#include <assert.h>
#include <cmath>
#include <sys/stat.h>

#include "../lib/vtk.cuh"
#include "../lib/n2n.cuh"
// #include "../lib/lattice.cuh"


const float R_MAX = 1;
const float R_MIN = 0.5;
const int N_CELLS = 1000;
const int N_TIME_STEPS = 200;
const float DELTA_T = 0.01;

__device__ __managed__ float3 X[N_CELLS];


// Smooth transition from step(x < 0) = 0 to step(x > 0) = 1 over dx
__device__ float step(float x) {
    float dx = 0.1;
    x = __saturatef((x + dx/2)/dx);
    return x*x*(3 - 2*x);
}

// Squeeze against floor
__global__ void squeeze(float3 X[], float time_step) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        X[i].z += 10*step(-2 - X[i].z)*DELTA_T; // Floor
        if ((time_step >= 20) && (time_step <= 100)) {
            X[i].z -= 10*step(X[i].z - (2 - (time_step - 20)/60))*DELTA_T;
        }
    }
}


__device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (dist > 1e-7) {
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
    float r_sphere = pow(N_CELLS/0.75, 1./3)*R_MIN/2; // Sphere packing
    for (int i = 0; i < N_CELLS; i++) {
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
        sprintf(file_name, "output/round_up_%03i.vtk", time_step);
        write_positions(file_name, N_CELLS, X);

        if (time_step < N_TIME_STEPS) {
            euler_step(DELTA_T, N_CELLS, X);
            squeeze<<<(N_CELLS + 16 - 1)/16, 16>>>(X, time_step);
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
