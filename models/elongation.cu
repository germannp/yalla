// Simulating a layer.
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <curand_kernel.h>

#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 5000;
const float R_CONN = 1.5;
const int N_CONNECTIONS = 2500;
const int N_TIME_STEPS = 500;
const float DELTA_T = 0.2;

__device__ __managed__ Solution<float4, N_CELLS, LatticeSolver> X;
__device__ __managed__ int cell_type[N_CELLS];

__device__ __managed__ int connections[N_CONNECTIONS][2];
__device__ __managed__ curandState rand_states[N_CONNECTIONS];


__device__ float4 cubic_w_diffusion(float4 Xi, float4 Xj, int i, int j) {
    float4 dF = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z, Xi.w - Xj.w};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    if (i != j) {
        float F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
        dF.x = r.x*F/dist;
        dF.y = r.y*F/dist;
        dF.z = r.z*F/dist;
        float D = dist < R_MAX ? 0.1 : 0;
        dF.w = - r.w*D;
    } else {
        assert(Xi.w >= 0);
        dF.w = cell_type[i] - 0.01*Xi.w;
    }
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float4> p_potential = cubic_w_diffusion;


__device__ float4 count_neighbours(float4 Xi, float4 Xj, int i, int j) {
    float4 dF = {0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float4 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z, Xi.w - Xj.w};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
    dF.w = dist < R_MAX ? 1 : 0;
    if (i == 0) cell_type[j] = dF.w;
    return dF;
}

__device__ __managed__ nhoodint<float4> p_count = count_neighbours;


__global__ void setup_rand_states() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CONNECTIONS) curand_init(1337, i, 0, &rand_states[i]);
}


__global__ void update_connections(const int* __restrict__ cell_id,
        const int* __restrict__ cube_id, const int* __restrict__ cube_start,
        const int* __restrict__ cube_end) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CONNECTIONS) return;

    int j = (int)(curand_uniform(&rand_states[i])*N_CELLS);
    int rand_cube = cube_id[j]
        +  (int)(curand_uniform(&rand_states[i])*3) - 1
        + ((int)(curand_uniform(&rand_states[i])*3) - 1)*LATTICE_SIZE
        + ((int)(curand_uniform(&rand_states[i])*3) - 1)*LATTICE_SIZE*LATTICE_SIZE;
    int cells_in_cube = cube_end[rand_cube] - cube_start[rand_cube];
    if (cells_in_cube < 1) return;

    int k = cube_start[rand_cube]
        + (int)(curand_uniform(&rand_states[i])*cells_in_cube);
    float4 r = {X[cell_id[j]].x - X[cell_id[k]].x, X[cell_id[j]].y - X[cell_id[k]].y,
        X[cell_id[j]].z - X[cell_id[k]].z, X[cell_id[j]].w - X[cell_id[k]].w};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if ((j != k) && (cell_type[cell_id[j]] != 1) && (cell_type[cell_id[k]] != 1)
            && (dist < R_CONN) && (fabs(r.w/(X[cell_id[j]].w + X[cell_id[k]].w)) > 0.2)) {
        connections[i][0] = cell_id[j];
        connections[i][1] = cell_id[k];
    }
}

__global__ void intercalate(const __restrict__ float4* X, float4* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_CONNECTIONS) return;

    if (connections[i][0] == connections[i][1]) return;

    float4 Xi = X[connections[i][0]];
    float4 Xj = X[connections[i][1]];
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);

    dX[connections[i][0]].x -= r.x/dist;
    dX[connections[i][0]].y -= r.y/dist;
    dX[connections[i][0]].z -= r.z/dist;
    dX[connections[i][1]].x += r.x/dist;
    dX[connections[i][1]].y += r.y/dist;
    dX[connections[i][1]].z += r.z/dist;
}

void intercalation(const float4* __restrict__ X, float4* dX) {
    intercalate<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>(X, dX);
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(N_CELLS, 0.733333, X);
    for (int i = 0; i < N_CELLS; i++) {
        X[i].x = fabs(X[i].x);
        X[i].y = X[i].y/1.5;
        X[i].w = 0;
        cell_type[i] = 0;
    }
    for (int i = 0; i < N_CONNECTIONS; i++) {
        connections[i][0] = 0;
        connections[i][1] = 0;
    }
    setup_rand_states<<<(N_CONNECTIONS + 128 - 1)/128, 128>>>();
    cudaDeviceSynchronize();

    // Relax
    VtkOutput relax_output("relaxation");
    for (int time_step = 0; time_step <= 200; time_step++) {
        X.step(DELTA_T, p_potential);
        relax_output.print_progress();
    }
    relax_output.print_done();

    // Find epithelium
    X.step(1, p_count);
    // X.z_order(N_CELLS, 2.);
    for (int i = 0; i < N_CELLS; i++) {
        cell_type[i] = X[i].w < 12 ? 1*(X[i].x > 0) : 0;
        X[i].w = 0;
    }

    // Simulate diffusion & intercalation
    VtkOutput sim_output("elongation");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        sim_output.write_positions(N_CELLS, X);
        sim_output.write_connections(N_CONNECTIONS, connections);
        sim_output.write_type(N_CELLS, cell_type);
        sim_output.write_field(N_CELLS, "w", X);
        if (time_step < N_TIME_STEPS) {
            // X.step(DELTA_T, p_potential);
            X.step(DELTA_T, p_potential, intercalation);
            X.build_lattice(N_CELLS, R_CONN);
            update_connections<<<(N_CONNECTIONS + 32 - 1)/32, 32>>>(X.cell_id, X.cube_id,
                X.cube_start, X.cube_end);
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
