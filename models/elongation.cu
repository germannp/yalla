// Simulating a layer.
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_MAX = 61000;
const float R_CONN = 1.5;
const float CONNS_P_CELL = 1;
const int N_TIME_STEPS = 500;
const float DELTA_T = 0.2;

__device__ __managed__ Solution<float4, N_MAX, LatticeSolver> X;
__device__ __managed__ int cell_type[N_MAX];
__device__ __managed__ int n_cells = 5000;

__device__ __managed__ int connections[(int)(N_MAX*CONNS_P_CELL)][2];
__device__ curandState rand_states[(int)(N_MAX*CONNS_P_CELL)];


__device__ float4 cubic_w_diffusion(float4 Xi, float4 Xj, int i, int j) {
    float4 dF = {0.0f, 0.0f, 0.0f, 0.0f};
    if (i != j) {
        float4 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z, Xi.w - Xj.w};
        float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), R_MAX);
        float F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
        dF.x = r.x*F/dist*(Xi.x > 0);
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
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF.w = dist < R_MAX ? 1 : 0;
    return dF;
}

__device__ __managed__ nhoodint<float4> p_count = count_neighbours;


__global__ void setup_rand_states() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_MAX*CONNS_P_CELL) curand_init(1337, i, 0, &rand_states[i]);
}


__global__ void update_connections(const int* __restrict__ cell_id,
        const int* __restrict__ cube_id, const int* __restrict__ cube_start,
        const int* __restrict__ cube_end) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*CONNS_P_CELL) return;

    int j = (int)(curand_uniform(&rand_states[i])*n_cells);
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
            // && (fabs(r.x/dist) < 0.2) && (j != k) && (dist < 2)) {
        connections[i][0] = cell_id[j];
        connections[i][1] = cell_id[k];
    }
}

__global__ void intercalate(const __restrict__ float4* X, float4* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*CONNS_P_CELL) return;

    if (connections[i][0] == connections[i][1]) return;

    float4 Xi = X[connections[i][0]];
    float4 Xj = X[connections[i][1]];
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);

    atomicAdd(&dX[connections[i][0]].x, -r.x/dist/2);
    atomicAdd(&dX[connections[i][0]].y, -r.y/dist/2);
    atomicAdd(&dX[connections[i][0]].z, -r.z/dist/2);
    atomicAdd(&dX[connections[i][1]].x, r.x/dist/2);
    atomicAdd(&dX[connections[i][1]].y, r.y/dist/2);
    atomicAdd(&dX[connections[i][1]].z, r.z/dist/2);
}

void intercalation(const float4* __restrict__ X, float4* dX) {
    intercalate<<<(n_cells*CONNS_P_CELL + 32 - 1)/32, 32>>>(X, dX);
    cudaDeviceSynchronize();
}


void proliferate(float rate, float mean_distance) {
    assert(rate*n_cells <= N_MAX);
    int i; float phi, theta;
    for (int j = 1; j < rate*n_cells; j++) {
        i = (int)(rand()/(RAND_MAX + 1.)*n_cells);
        phi = rand()/(RAND_MAX + 1.)*M_PI;
        theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        X[n_cells].x = X[i].x + mean_distance/2*sinf(theta)*cosf(phi);
        X[n_cells].y = X[i].y + mean_distance/2*sinf(theta)*sinf(phi);
        X[n_cells].z = X[i].z + mean_distance/2*cosf(theta);
        X[n_cells].w = X[i].w/2;
        X[i].w = X[i].w/2;
        cell_type[n_cells] = cell_type[i];
        n_cells++;
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(n_cells, 0.733333, X);
    for (int i = 0; i < n_cells; i++) {
        X[i].x = fabs(X[i].x);
        X[i].y = X[i].y/1.5;
        X[i].w = 0;
        cell_type[i] = 0;
    }
    for (int i = 0; i < N_MAX*CONNS_P_CELL; i++) {
        connections[i][0] = 0;
        connections[i][1] = 0;
    }
    setup_rand_states<<<(N_MAX*CONNS_P_CELL + 128 - 1)/128, 128>>>();
    cudaDeviceSynchronize();

    // Relax
    VtkOutput relax_output("relaxation");
    for (int time_step = 0; time_step <= 200; time_step++) {
        X.step(DELTA_T, p_potential, n_cells);
        relax_output.print_progress();
    }
    relax_output.print_done();

    // Find epithelium
    X.step(1, p_count, n_cells);
    // X.z_order(n_cells, 2.);
    for (int i = 0; i < n_cells; i++) {
        cell_type[i] = X[i].w < 12 ? 1*(X[i].x > 0) : 0;
        X[i].w = 0;
    }

    // Simulate diffusion & intercalation
    VtkOutput sim_output("elongation");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        sim_output.write_positions(n_cells, X);
        sim_output.write_connections(n_cells*CONNS_P_CELL, connections);
        sim_output.write_type(n_cells, cell_type);
        sim_output.write_field(n_cells, "w", X);
        if (time_step == N_TIME_STEPS) return 0;

        // X.step(DELTA_T, p_potential, n_cells);
        X.step(DELTA_T, p_potential, intercalation, n_cells);
        proliferate(0.005, 0.733333);
        X.build_lattice(n_cells, R_CONN);
        update_connections<<<(n_cells*CONNS_P_CELL + 32 - 1)/32, 32>>>(X.cell_id,
            X.cube_id, X.cube_start, X.cube_end);
        cudaDeviceSynchronize();
    }
}
