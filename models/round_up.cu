// Simulating rounding up.
#include <assert.h>
#include <cmath>
#include <sys/stat.h>

#include "../lib/sphere.cuh"
#include "../lib/vtk.cuh"
// #include "../lib/n2n.cuh"
#include "../lib/lattice.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 1000;
const float DELTA_T = 0.0025;

__device__ __managed__ float3 X[N_CELLS], dX[N_CELLS];


// Smooth transition from step(x < 0) = 0 to step(x > 0) = 1 over dx
__device__ float step(float x) {
    float dx = 0.1;
    x = __saturatef((x + dx/2)/dx);
    return x*x*(3 - 2*x);
}

// Squeeze against floor
__global__ void squeeze(float3 X[], float time_step) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float time = time_step*DELTA_T;
    if (i < N_CELLS) {
        X[i].z += 10*step(-2 - X[i].z)*DELTA_T; // Floor
        if ((time >= 0.1) && (time <= 0.5)) {
            X[i].z -= 10*step(X[i].z - (2 - (time - 0.1)/0.3))*DELTA_T;
        }
    }
}


__device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (dist > 1e-7) {
        int n = 2;
        float strength = 100;
        float F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
            + strength*powf(R_MAX - dist, n);
        dF.x = r.x*F/dist;
        dF.y = r.y*F/dist;
        dF.z = r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, 0.733333, X);

    // Integrate cell positions
    VtkOutput output("round_up");
    for (int time_step = 0; time_step*DELTA_T <= 1; time_step++) {
        output.write_positions(N_CELLS, X);

        if (time_step*DELTA_T <= 1) {
            euler_step(DELTA_T, N_CELLS, X, dX);
            squeeze<<<(N_CELLS + 16 - 1)/16, 16>>>(X, time_step);
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
