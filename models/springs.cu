// Integrate N-body problem with springs between all bodies. Parallelization
// after http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html.

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "../lib/vtk.cu"


const float L_0 = 0.5; // Relaxed spring length
const uint N_BODIES = 800;
const uint N_TIME_STEPS = 100;
const uint TILE_SIZE = 32;

__device__ __managed__ float3 X[N_BODIES];

__device__ float3 body_body_force(float3 Xi, float3 Xj) {
    float3 r;
    float3 dF = {0.0f, 0.0f, 0.0f};
    r.x = Xj.x - Xi.x;
    r.y = Xj.y - Xi.y;
    r.z = Xj.z - Xi.z;
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > 1e-8) {
        dF.x += r.x*(dist - L_0)/dist;
        dF.y += r.y*(dist - L_0)/dist;
        dF.z += r.z*(dist - L_0)/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

// Calculate new X one thread per body, to TILE_SIZE other bodies at a time
__global__ void integrate_step() {
    __shared__ float3 shX[TILE_SIZE];
    int body_idx = blockIdx.x*blockDim.x + threadIdx.x;
    float3 Xi = X[body_idx];
    float3 Fi = {0.0f, 0.0f, 0.0f};
    for (int tile_start = 0; tile_start < N_BODIES; tile_start += TILE_SIZE) {
        int other_body_idx = tile_start + threadIdx.x;
        shX[threadIdx.x] = X[other_body_idx];
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            float3 dF = body_body_force(Xi, shX[i]);
            Fi.x += dF.x;
            Fi.y += dF.y;
            Fi.z += dF.z;
        }
    }
    X[body_idx].x = Xi.x + Fi.x*0.001;
    X[body_idx].y = Xi.y + Fi.y*0.001;
    X[body_idx].z = Xi.z + Fi.z*0.001;
}

int main(int argc, const char* argv[]) {
    assert(N_BODIES % TILE_SIZE == 0);

    // Prepare initial state
    float r_max = pow(N_BODIES/0.75, 1./3)*L_0/2; // Sphere packing
    for (int i = 0; i < N_BODIES; i++) {
        float r = r_max*rand()/(RAND_MAX + 1.);
        float theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        float phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        X[i].x = r*sin(theta)*sin(phi);
        X[i].y = r*cos(theta)*sin(phi);
        X[i].z = r*cos(phi);
    }

    // Integrate body positions
    mkdir("output", 755);
    int n_blocks = (N_BODIES + TILE_SIZE - 1)/TILE_SIZE; // ceil int div.
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        std::stringstream file_name;
        file_name << "output/springs_" << time_step << ".vtk";
        write_positions(file_name.str().c_str(), N_BODIES, X);

        if (time_step < N_TIME_STEPS) {
            integrate_step<<<n_blocks, TILE_SIZE>>>();
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
