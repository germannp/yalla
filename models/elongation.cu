// Simulate elongation of semisphere
#include <assert.h>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"
#include "../lib/epithelium.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_MAX = 61000;
const auto R_CONN = 1.5;
const auto CONNS_P_CELL = 1;
const auto N_TIME_STEPS = 500;
const auto DELTA_T = 0.2;
enum CELL_TYPES {MESENCHYME, STRECHED_EPI, EPITHELIUM};

__device__ __managed__ CELL_TYPES cell_type[N_MAX];
__device__ __managed__ auto n_cells = 5000;
__device__ __managed__ int connections[static_cast<int>(N_MAX*CONNS_P_CELL)][2];
__device__ curandState rand_states[static_cast<int>(N_MAX*CONNS_P_CELL)];


MAKE_DTYPE(lbcell, x, y, z, w, phi, theta);

__device__ __managed__ Solution<lbcell, N_MAX, LatticeSolver> X;


__device__ lbcell cubic_w_diffusion(lbcell Xi, lbcell Xj, int i, int j) {
    libcell dF {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) {
        assert(Xi.w >= 0);
        dF.w = (cell_type[i] > MESENCHYME) - 0.01*Xi.w;
        return dF;
    }

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + (R_MAX - dist)*(R_MAX - dist);
    dF.x = r.x*F/dist*(Xi.x > 0);
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    auto D = dist < R_MAX ? 0.1 : 0;
    dF.w = - r.w*D;
    assert(dF.x == dF.x);  // For NaN f != f.

    if (cell_type[i] == MESENCHYME or cell_type[j] == MESENCHYME) return dF;

    if (dist < 0.733333) cell_type[i] = EPITHELIUM;
    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ __managed__ auto d_potential = cubic_w_diffusion;


__device__ lbcell count_neighbours(lbcell Xi, lbcell Xj, int i, int j) {
    libcell dF {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF.w = dist < R_MAX ? 1 : 0;
    return dF;
}

__device__ __managed__ auto d_count = count_neighbours;


__global__ void setup_rand_states() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_MAX*CONNS_P_CELL) curand_init(1337, i, 0, &rand_states[i]);
}


__global__ void update_connections(const int* __restrict__ cell_id,
        const int* __restrict__ cube_id, const int* __restrict__ cube_start,
        const int* __restrict__ cube_end) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*CONNS_P_CELL) return;

    auto j = static_cast<int>(curand_uniform(&rand_states[i])*n_cells);
    auto rand_cube = cube_id[j]
        +  static_cast<int>(curand_uniform(&rand_states[i])*3) - 1
        + (static_cast<int>(curand_uniform(&rand_states[i])*3) - 1)*LATTICE_SIZE
        + (static_cast<int>(curand_uniform(&rand_states[i])*3) - 1)*LATTICE_SIZE*LATTICE_SIZE;
    auto cells_in_cube = cube_end[rand_cube] - cube_start[rand_cube];
    if (cells_in_cube < 1) return;

    auto k = cube_start[rand_cube]
        + static_cast<int>(curand_uniform(&rand_states[i])*cells_in_cube);
    auto r = X[cell_id[j]] - X[cell_id[k]];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if ((j != k) and (cell_type[cell_id[j]] == MESENCHYME)
            and (cell_type[cell_id[k]] == MESENCHYME)
            and (dist < R_CONN) and (fabs(r.w/(X[cell_id[j]].w + X[cell_id[k]].w)) > 0.2)) {
            // and (fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
        connections[i][0] = cell_id[j];
        connections[i][1] = cell_id[k];
    }
}

__global__ void intercalate(const lbcell* __restrict__ X, lbcell* dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*CONNS_P_CELL) return;

    if (connections[i][0] == connections[i][1]) return;

    auto Xi = X[connections[i][0]];
    auto Xj = X[connections[i][1]];
    auto r = float3{Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);

    atomicAdd(&dX[connections[i][0]].x, -r.x/dist/2);
    atomicAdd(&dX[connections[i][0]].y, -r.y/dist/2);
    atomicAdd(&dX[connections[i][0]].z, -r.z/dist/2);
    atomicAdd(&dX[connections[i][1]].x, r.x/dist/2);
    atomicAdd(&dX[connections[i][1]].y, r.y/dist/2);
    atomicAdd(&dX[connections[i][1]].z, r.z/dist/2);
}

void intercalation(const lbcell* __restrict__ X, lbcell* dX) {
    intercalate<<<(n_cells*CONNS_P_CELL + 32 - 1)/32, 32>>>(X, dX);
    cudaDeviceSynchronize();
}


__global__ void proliferate(float rate, float mean_distance) {
    assert(rate*n_cells <= N_MAX);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (cell_type[i] == EPITHELIUM) {
        cell_type[i] = STRECHED_EPI;
        return;
    }

    if (cell_type[i] == MESENCHYME) {
        auto r = curand_uniform(&rand_states[i]);
        if (r > rate) return;
    }

    auto n = atomicAdd(&n_cells, 1);
    auto phi = curand_uniform(&rand_states[i])*M_PI;
    auto theta = curand_uniform(&rand_states[i])*2*M_PI;
    X[n].x = X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    X[n].y = X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    X[n].z = X[i].z + mean_distance/4*cosf(theta);
    X[n].w = X[i].w/2;
    X[i].w = X[i].w/2;
    X[n].phi = X[i].phi;
    X[n].theta = X[i].theta;
    cell_type[n] = cell_type[i] == MESENCHYME ? MESENCHYME : STRECHED_EPI;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(0.733333, X, n_cells);
    for (auto i = 0; i < n_cells; i++) {
        X[i].x = fabs(X[i].x);
        X[i].y = X[i].y/1.5;
        X[i].w = 0;
        cell_type[i] = MESENCHYME;
    }
    for (auto i = 0; i < N_MAX*CONNS_P_CELL; i++) {
        connections[i][0] = 0;
        connections[i][1] = 0;
    }
    setup_rand_states<<<(N_MAX*CONNS_P_CELL + 128 - 1)/128, 128>>>();
    cudaDeviceSynchronize();

    // Relax
    VtkOutput relax_output("relaxation");
    for (auto time_step = 0; time_step <= 200; time_step++) {
        X.step(DELTA_T, d_potential, n_cells);
        relax_output.print_progress();
    }
    relax_output.print_done();

    // Find epithelium
    X.step(1, d_count, n_cells);
    // X.z_order(n_cells, 2.);
    for (auto i = 0; i < n_cells; i++) {
        if (X[i].w < 12 and X[i].x > 0) {
            cell_type[i] = STRECHED_EPI;
            auto dist = sqrtf(X[i].x*X[i].x + X[i].y*X[i].y + X[i].z*X[i].z);
            X[i].phi = atan2(X[i].y, X[i].x);
            X[i].theta = acosf(X[i].z/dist);
        } else {
            X[i].phi = 0;
            X[i].theta = 0;
        }
        X[i].w = 0;
    }

    // Simulate diffusion & intercalation
    VtkOutput sim_output("elongation");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        sim_output.write_positions(X, n_cells);
        sim_output.write_connections(connections, n_cells*CONNS_P_CELL);
        sim_output.write_type(cell_type, n_cells);
        // sim_output.write_polarity(X, n_cells);
        sim_output.write_field(X, n_cells, "Wnt");
        if (time_step == N_TIME_STEPS) return 0;

        // X.step(DELTA_T, d_potential, n_cells);
        X.step(DELTA_T, d_potential, intercalation, n_cells);
        proliferate<<<(n_cells + 128 - 1)/128, 128>>>(0.005, 0.733333);
        cudaDeviceSynchronize();
        X.build_lattice(n_cells, R_CONN);
        update_connections<<<(n_cells*CONNS_P_CELL + 32 - 1)/32, 32>>>(X.cell_id,
            X.cube_id, X.cube_start, X.cube_end);
        cudaDeviceSynchronize();
    }
}
