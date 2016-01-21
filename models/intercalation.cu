// Simulate intercalating cells
#include <assert.h>
#include <cmath>
#include <sys/stat.h>
#include <curand_kernel.h>

#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"
#include "../lib/n2n.cuh"
// #include "../lib/lattice.cuh"


const float R_MAX = 1;
const float R_MIN = 0.5;
const int N_CELLS = 500;
const int N_CONNECTIONS = 250;
const int N_TIME_STEPS = 1000;
const float DELTA_T = 0.1;

__device__ __managed__ float3 X[N_CELLS], dX[N_CELLS];
__device__ __managed__ int connections[N_CONNECTIONS][2];
__device__ __managed__ curandState rand_states[N_CONNECTIONS];

__global__ void setup_rand_states() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) curand_init(1337, i, 0, &rand_states[i]);
}


__device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (dist > 1e-7) {
        float F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
        dF.x = r.x*F/dist;
        dF.y = r.y*F/dist;
        dF.z = r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}


__global__ void intercalate() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CONNECTIONS) {
        float3 Xi = X[connections[i][0]];
        float3 Xj = X[connections[i][1]];
        float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        X[connections[i][0]].x -= r.x/dist*DELTA_T/5;
        X[connections[i][0]].y -= r.y/dist*DELTA_T/5;
        X[connections[i][0]].z -= r.z/dist*DELTA_T/5;
        X[connections[i][1]].x += r.x/dist*DELTA_T/5;
        X[connections[i][1]].y += r.y/dist*DELTA_T/5;
        X[connections[i][1]].z += r.z/dist*DELTA_T/5;

        int j = (int)(curand_uniform(&rand_states[i])*N_CELLS);
        int k = (int)(curand_uniform(&rand_states[i])*N_CELLS);
        r = {X[j].x - X[k].x, X[j].y - X[k].y, X[j].z - X[k].z};
        dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if ((fabs(r.x/dist) < 0.2) && (j != k) && (dist < 2)) {
            connections[i][0] = j;
            connections[i][1] = k;
        }
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, R_MIN, X);
    setup_rand_states<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>();
    cudaDeviceSynchronize();
    int i = 0;
    while (i < N_CONNECTIONS) {
        int j = (int)(rand()/(RAND_MAX + 1.)*N_CELLS);
        int k = (int)(rand()/(RAND_MAX + 1.)*N_CELLS);
        float3 r = {X[j].x - X[k].x, X[j].y - X[k].y, X[j].z - X[k].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if ((fabs(r.x/dist) < 0.2) && (j != k) && (dist < 2)) {
            connections[i][0] = j;
            connections[i][1] = k;
            i++;
        }
    }
    // Integrate cell positions
    VtkOutput output("intercalation");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        output.write_connections(N_CONNECTIONS, connections);
        if (time_step < N_TIME_STEPS) {
            euler_step(DELTA_T, N_CELLS, X, dX);
            intercalate<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>();
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
