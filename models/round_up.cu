// Simulate rounding up
#include <assert.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_CELLS = 100;
const auto DELTA_T = 0.005;

__device__ __managed__ Solution<float3, N_CELLS, LatticeSolver> X;
__device__ __managed__ auto time_step = 0;


__device__ float3 clipped_polynomial(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto n = 2;
    auto strength = 100;
    auto F = strength*n*(R_MIN - dist)*powf(R_MAX - dist, n - 1)
        + strength*powf(R_MAX - dist, n);
    // auto F = strength*(fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0)/2);
    dF = r*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ auto d_potential = clipped_polynomial;


// Smooth transition from step(x < 0) = 0 to step(x > 0) = 1 over dx
__device__ float step(float x) {
    auto dx = 0.1;
    x = __saturatef((x + dx/2)/dx);
    return x*x*(3 - 2*x);
}

__global__ void squeeze_kernel(const float3* __restrict__ X, float3* dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;

    auto time = time_step*DELTA_T;
    dX[i].z += 10*step(-2 - X[i].z);  // Floor
    if ((time >= 0.1) and (time <= 0.5)) {
        dX[i].z -= 10*step(X[i].z - (2 - (time - 0.1)/0.3));
    }
}

void squeeze_to_floor(const float3* __restrict__ X, float3* dX) {
    squeeze_kernel<<<(N_CELLS + 16 - 1)/16, 16>>>(X, dX);
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_circle(0.733333, X);
    // uniform_sphere(0.733333, X);

    // Integrate cell positions
    VtkOutput output("round_up");
    for (time_step = 0; time_step*DELTA_T <= 1; time_step++) {
        output.write_positions(X);
        if (time_step*DELTA_T == 1) return 0;

        X.step(DELTA_T, d_potential, squeeze_to_floor);
    }
}
