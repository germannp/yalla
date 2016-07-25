// Simulate growing mesenchyme envelopped by epithelium
#include <assert.h>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"
#include "../lib/epithelium.cuh"


const auto R_MAX = 1;
const auto MEAN_DIST = 0.75;
const auto RATE = 0.006;
const auto N_MAX = 5000;
const auto N_TIME_STEPS = 500;
const auto DELTA_T = 0.2;
enum CELL_TYPES {MESENCHYME, EPITHELIUM};

Solution<pocell, N_MAX, LatticeSolver> bolls;
__device__ __managed__ CELL_TYPES cell_type[N_MAX];
__device__ __managed__ int n_mes_neighbrs[N_MAX];
__device__ __managed__ int n_epi_neighbrs[N_MAX];
__device__ curandState rand_states[N_MAX];


__device__ pocell cubic_w_polarity(pocell Xi, pocell Xj, int i, int j) {
    pocell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
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

    if (cell_type[j] == MESENCHYME) n_mes_neighbrs[i] += 1;
    else n_epi_neighbrs[i] += 1;

    if (cell_type[i] == MESENCHYME or cell_type[j] == MESENCHYME) return dF;

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ auto d_cubic_w_polarity = &cubic_w_polarity;
auto h_cubic_w_polarity = get_device_object(d_cubic_w_polarity, 0);


__global__ void reset_n_neighbrs(int n_cells) {  // TODO: Use thrust?
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) n_mes_neighbrs[i] = 0;
    if (i < n_cells) n_epi_neighbrs[i] = 0;
}

__global__ void setup_rand_states() {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N_MAX) curand_init(1337, i, 0, &rand_states[i]);
}

__global__ void proliferate(float rate, float mean_distance, pocell* d_X, int* n_cells) {
    assert(rate* *n_cells <= N_MAX);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *n_cells) return;

    switch (cell_type[i]) {
        case MESENCHYME: {
            auto r = curand_uniform(&rand_states[i]);
            if (r > rate) return;
        }
        case EPITHELIUM: {
            if (n_epi_neighbrs[i] > n_mes_neighbrs[i]) return;
        }
    }

    auto n = atomicAdd(n_cells, 1);
    auto phi = curand_uniform(&rand_states[i])*M_PI;
    auto theta = curand_uniform(&rand_states[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].phi = d_X[i].phi;
    d_X[n].theta = d_X[i].theta;
    cell_type[n] = cell_type[i] == MESENCHYME ? MESENCHYME : EPITHELIUM;
    n_mes_neighbrs[n] = 0;
    n_epi_neighbrs[n] = 0;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    bolls.set_n(200);
    uniform_sphere(MEAN_DIST, bolls);
    for (auto i = 0; i < bolls.get_n(); i++) cell_type[i] = MESENCHYME;
    setup_rand_states<<<(N_MAX + 128 - 1)/128, 128>>>();

    // Relax
    for (auto time_step = 0; time_step <= 500; time_step++) {
        reset_n_neighbrs<<<(bolls.get_n() + 128 - 1)/128, 128>>>(bolls.get_n());
        bolls.step(DELTA_T, h_cubic_w_polarity);
    }

    // Find epithelium
    bolls.memcpyDeviceToHost();
    for (auto i = 0; i < bolls.get_n(); i++) {
        if (n_mes_neighbrs[i] < 12*2) {  // 2nd order solver
            cell_type[i] = EPITHELIUM;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x + bolls.h_X[i].y*bolls.h_X[i].y
                + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
        } else {
            bolls.h_X[i].phi = 0;
            bolls.h_X[i].theta = 0;
        }
    }
    bolls.memcpyHostToDevice();

    // Simulate growth
    VtkOutput sim_output("passive_growth");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        sim_output.write_positions(bolls);
        sim_output.write_type(cell_type);
        sim_output.write_polarity(bolls);
        reset_n_neighbrs<<<(bolls.get_n() + 128 - 1)/128, 128>>>(bolls.get_n());
        bolls.step(DELTA_T, h_cubic_w_polarity);
        proliferate<<<(bolls.get_n() + 128 - 1)/128, 128>>>(RATE*(time_step > 100),
            MEAN_DIST, bolls.d_X, bolls.d_n);
    }

    return 0;
}
