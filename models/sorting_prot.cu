// Simulate cell sorting by protrusions
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1.f;
const auto R_MIN = 0.5f;
const auto N_CELLS = 200u;
const auto N_CONNECTIONS = N_CELLS*5;
const auto N_TIME_STEPS = 300u;
const auto DELTA_T = 0.05;

__device__ __managed__ Solution<float3, N_CELLS, LatticeSolver> X;
__device__ __managed__ int connections[N_CONNECTIONS][2];
__device__ curandState rand_states[N_CONNECTIONS];


__device__ float3 cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF = r*F/dist;
    return dF;
}

__device__ __managed__ auto d_cubic = cubic;


__global__ void setup_rand_states() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) curand_init(1337, i, 0, &rand_states[i]);
}

__global__ void intercalate(const float3* __restrict__ X, float3* dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CONNECTIONS) return;

    auto j = connections[i][0];
    auto k = connections[i][1];
    if (j == k) return;

    auto r = X[j] - X[k];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    atomicAdd(&dX[connections[i][0]].x, -r.x/dist/5);
    atomicAdd(&dX[connections[i][0]].y, -r.y/dist/5);
    atomicAdd(&dX[connections[i][0]].z, -r.z/dist/5);
    atomicAdd(&dX[connections[i][1]].x, r.x/dist/5);
    atomicAdd(&dX[connections[i][1]].y, r.y/dist/5);
    atomicAdd(&dX[connections[i][1]].z, r.z/dist/5);
}

void intercalation(const float3* __restrict__ X, float3* dX) {
    intercalate<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>(X, dX);
    cudaDeviceSynchronize();
}

__global__ void update_connections() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CONNECTIONS) return;

    auto r = curand_uniform(&rand_states[i]);
    auto j = connections[i][0];
    auto k = connections[i][1];
    if ((j < N_CELLS/2) and (k < N_CELLS/2)) {
        if (r > 0.05) return;
    } else if ((j > N_CELLS/2) and (k > N_CELLS/2)) {
        if (r > 0.25) return;
    } else {
        if (r > 0.75) return;
    }

    auto new_j = static_cast<int>(curand_uniform(&rand_states[i])*N_CELLS);
    auto new_k = static_cast<int>(curand_uniform(&rand_states[i])*N_CELLS);
    if (new_j == new_k) return;

    auto dx = X[new_j] - X[new_k];
    auto dist = sqrtf(dx.x*dx.x + dx.y*dx.y + dx.z*dx.z);
    if (dist > 2) return;

    connections[i][0] = new_j;
    connections[i][1] = new_k;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(R_MIN, X);
    int cell_type[N_CELLS];
    for (auto i = 0; i < N_CELLS; i++) {
        cell_type[i] = (i < N_CELLS/2) ? 0 : 1;
    }
    for (auto i = 0; i < N_CONNECTIONS; i++) {
        connections[i][0] = 0;
        connections[i][1] = 0;
    }
    setup_rand_states<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>();
    cudaDeviceSynchronize();


    // Integrate cell positions
    VtkOutput output("sorting");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_connections(connections, N_CONNECTIONS);
        output.write_type(cell_type);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, d_cubic, intercalation);
        update_connections<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>();
        cudaDeviceSynchronize();
    }
}
