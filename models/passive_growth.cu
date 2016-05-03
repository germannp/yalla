// Simulate growing mesenchyme envelopped by epithelium
#include <assert.h>
#include <curand_kernel.h>
#include <cmath>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"
#include "../lib/epithelium.cuh"


const float R_MAX = 1;
const float RATE = 0.006;
const float MEAN_DIST = 0.75;
const int N_MAX = 5000;
const int N_TIME_STEPS = 500;
const float DELTA_T = 0.2;
enum CELL_TYPES {MESENCHYME, EPITHELIUM};

__device__ __managed__ Solution<pocell, N_MAX, LatticeSolver> X;
__device__ __managed__ CELL_TYPES cell_type[N_MAX];
__device__ __managed__ int n_neighbrs[N_MAX];
__device__ __managed__ int n_cells = 200;
__device__ curandState rand_states[N_MAX];


__device__ pocell cubic_w_polarity(pocell Xi, pocell Xj, int i, int j) {
    pocell dF = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    float F;
    if (cell_type[i] == cell_type[j]) {
        F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0)/2;
    } else {
        F = fmaxf(0.8 - dist, 0)*2 - fmaxf(dist - 0.9, 0)/2;
    }
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.

    if (cell_type[i] == MESENCHYME) {
        n_neighbrs[i] += 1;
        return dF;
    }
    if (cell_type[j] == MESENCHYME) return dF;

    n_neighbrs[i] += 1;  // Count only EPITHELIUM neighbours for EPITHELIUM
    dF = dF + polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ __managed__ nhoodint<pocell> p_potential = cubic_w_polarity;


__global__ void reset_n_neighbrs() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) n_neighbrs[i] = 0;
}

__global__ void setup_rand_states() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_MAX) curand_init(1337, i, 0, &rand_states[i]);
}

__global__ void proliferate(float rate, float mean_distance) {
    assert(rate*n_cells <= N_MAX);
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    switch (cell_type[i]) {
        case MESENCHYME: {
            float r = curand_uniform(&rand_states[i]);
            if (r > rate) return;
            break;
        }
        case EPITHELIUM: {
            if (n_neighbrs[i] > 4*2) return;  // 2nd order solver
        }
    }

    int n = atomicAdd(&n_cells, 1);
    float phi = curand_uniform(&rand_states[i])*M_PI;
    float theta = curand_uniform(&rand_states[i])*2*M_PI;
    X[n].x = X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    X[n].y = X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    X[n].z = X[i].z + mean_distance/4*cosf(theta);
    X[n].phi = X[i].phi;
    X[n].theta = X[i].theta;
    cell_type[n] = cell_type[i];
    n_neighbrs[n] = 0;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(MEAN_DIST, X, n_cells);
    for (int i = 0; i < n_cells; i++) {
        cell_type[i] = MESENCHYME;
        n_neighbrs[i] = 0;
    }
    setup_rand_states<<<(N_MAX + 128 - 1)/128, 128>>>();
    cudaDeviceSynchronize();

    // Relax
    for (int time_step = 0; time_step <= 500; time_step++) {
        reset_n_neighbrs<<<(n_cells + 128 - 1)/128, 128>>>();
        cudaDeviceSynchronize();
        X.step(DELTA_T, p_potential, n_cells);
    }

    // Find epithelium
    for (int i = 0; i < n_cells; i++) {
        if (n_neighbrs[i] < 11*2) {  // 2nd order solver
            cell_type[i] = EPITHELIUM;
            float dist = sqrtf(X[i].x*X[i].x + X[i].y*X[i].y + X[i].z*X[i].z);
            X[i].phi = atan2(X[i].y, X[i].x);
            X[i].theta = acosf(X[i].z/dist);
        } else {
            X[i].phi = 0;
            X[i].theta = 0;
        }
    }

    // Simulate growth
    VtkOutput sim_output("passive_growth");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        sim_output.write_positions(X, n_cells);
        sim_output.write_type(cell_type, n_cells);
        sim_output.write_polarity(X, n_cells);
        if (time_step == N_TIME_STEPS) return 0;

        reset_n_neighbrs<<<(n_cells + 128 - 1)/128, 128>>>();
        cudaDeviceSynchronize();
        X.step(DELTA_T, p_potential, n_cells);
        proliferate<<<(n_cells + 128 - 1)/128, 128>>>(RATE, MEAN_DIST);
        cudaDeviceSynchronize();
    }
}
