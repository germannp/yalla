// Sorting based lattice for N-body problems with limited interaction, after
// http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
#include <assert.h>
#include <thrust/sort.h>

#include "util.cuh"


extern const float R_MAX;
extern __device__ float3 cell_cell_interaction(float3 Xi, float3 Xj, int i, int j);

const int MAX_N_CELLS = 1e6;
const int LATTICE_SIZE = 100;
const int N_CUBES = LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE;
const float CUBE_SIZE = 1;

__device__ __managed__ float max_F;
__device__ __managed__ int cube_id[MAX_N_CELLS];
__device__ __managed__ int cell_id[MAX_N_CELLS];
__device__ __managed__ int cube_start[LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE];
__device__ __managed__ int cube_end[LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE];


__global__ void compute_cube_ids(int N_CELLS, float3 X[]) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        float3 Xi = X[i];
        int id = (int)(
            (floor(Xi.x/CUBE_SIZE) + LATTICE_SIZE/2) +
            (floor(Xi.y/CUBE_SIZE) + LATTICE_SIZE/2)*LATTICE_SIZE +
            (floor(Xi.z/CUBE_SIZE) + LATTICE_SIZE/2)*LATTICE_SIZE*LATTICE_SIZE);
        assert(id >= 0);
        assert(id <= LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE);
        cube_id[i] = id;
        cell_id[i] = i;
    }
}

__global__ void reset_cube_start_and_end() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CUBES) {
        cube_start[i] = -1;
        cube_end[i] = -2;
    }
}

__global__ void compute_cube_start_and_end(int N_CELLS) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        int cube = cube_id[i];
        int prev = i > 0 ? cube_id[i - 1] : -1;
        if (cube != prev) cube_start[cube] = i;
        int next = i < N_CELLS ? cube_id[i + 1] : cube_id[i] + 1;
        if (cube != next) cube_end[cube] = i;
    }
}


__global__ void calculate_F(int N_CELLS, float3 X[], float3 F[]) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_CELLS) {
        int interacting_cubes[27];
        interacting_cubes[0] = cube_id[i] - 1;
        interacting_cubes[1] = cube_id[i];
        interacting_cubes[2] = cube_id[i] + 1;
        for (int j = 0; j < 3; j++) {
            interacting_cubes[j + 3] = interacting_cubes[j % 3] - LATTICE_SIZE;
            interacting_cubes[j + 6] = interacting_cubes[j % 3] + LATTICE_SIZE;
        }
        for (int j = 0; j < 9; j++) {
            interacting_cubes[j +  9] = interacting_cubes[j % 9] - LATTICE_SIZE*LATTICE_SIZE;
            interacting_cubes[j + 18] = interacting_cubes[j % 9] + LATTICE_SIZE*LATTICE_SIZE;
        }

        float3 dF, Fi  = {0.0f, 0.0f, 0.0f};
        float3 Xi = X[cell_id[i]];
        for (int j = 0; j < 27; j++) {
            int cube = interacting_cubes[j];
            for (int k = cube_start[cube]; k <= cube_end[cube]; k++) {
                float3 Xj = X[cell_id[k]];
                dF = cell_cell_interaction(Xi, Xj, cell_id[i], cell_id[k]);
                Fi.x += dF.x;
                Fi.y += dF.y;
                Fi.z += dF.z;
            }
        }
        F[cell_id[i]].x = Fi.x;
        F[cell_id[i]].y = Fi.y;
        F[cell_id[i]].z = Fi.z;
        atomicMax(&max_F, max(max(Fi.x, Fi.y), Fi.z));
    }
}

__global__ void integrate(float delta_t, int N_CELLS, float3 X[], float3 F[]) {
    int cell_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell_idx < N_CELLS) {
        X[cell_idx].x += F[cell_idx].x*delta_t;
        X[cell_idx].y += F[cell_idx].y*delta_t;
        X[cell_idx].z += F[cell_idx].z*delta_t;
    }
}


void dynamic_step(float delta_t, int N_CELLS, float3 X[], float3 F[]) {
    assert(LATTICE_SIZE % 2 == 0); // Needed?
    assert(N_CELLS <= MAX_N_CELLS);
    assert(R_MAX <= CUBE_SIZE);

    float left_t = delta_t;
    while (left_t > 0) {
        max_F = 0;
        compute_cube_ids<<<(N_CELLS + 16 - 1)/16, 16>>>(N_CELLS, X);
        cudaDeviceSynchronize();
        thrust::sort_by_key(cube_id, cube_id + N_CELLS, cell_id);
        reset_cube_start_and_end<<<(N_CUBES + 16 - 1)/16, 16>>>();
        cudaDeviceSynchronize();
        compute_cube_start_and_end<<<(N_CELLS + 16 - 1)/16, 16>>>(N_CELLS);
        cudaDeviceSynchronize();

        calculate_F<<<(N_CELLS + 16 - 1)/16, 16>>>(N_CELLS, X, F);
        cudaDeviceSynchronize();
        float dt = min(left_t, R_MAX/4/max_F);
        // printf("%f %f\n", left_t, dt);
        left_t -= dt;
        integrate<<<(N_CELLS + 16 - 1)/16, 16>>>(delta_t, N_CELLS, X, F);
        cudaDeviceSynchronize();
    }
}
