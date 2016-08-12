// Simulate rounding up
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_CELLS = 100;
const auto DELTA_T = 0.005;
auto time_step = 0;


__device__ float3 pairwise_interaction(float3 Xi, float3 Xj, int i, int j) {
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
    return dF;
}

#include "../lib/solvers.cuh"


// Smooth transition from step(x < 0) = 0 to step(x > 0) = 1 over dx
__device__ float step(float x) {
    auto dx = 0.1;
    x = __saturatef((x + dx/2)/dx);
    return x*x*(3 - 2*x);
}

__global__ void squeeze_kernel(const float3* __restrict__ bolls, float3* dX,
        int time_step) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;

    auto time = time_step*DELTA_T;
    dX[i].z += 10*step(-2 - bolls[i].z);  // Floor
    if ((time >= 0.1) and (time <= 0.5)) {
        dX[i].z -= 10*step(bolls[i].z - (2 - (time - 0.1)/0.3));
    }
}

void squeeze_to_floor(const float3* __restrict__ d_X, float3* d_dX) {
    squeeze_kernel<<<(N_CELLS + 16 - 1)/16, 16>>>(d_X, d_dX, time_step);
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<float3, N_CELLS, LatticeSolver> bolls;
    uniform_circle(0.733333, bolls);
    // uniform_sphere(0.733333, bolls);

    // Integrate cell positions
    VtkOutput output("round_up");
    for (time_step = 0; time_step*DELTA_T <= 1; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T, squeeze_to_floor);
        output.write_positions(bolls);
    }

    return 0;
}
