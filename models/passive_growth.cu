// Simulate growing mesenchyme envelopped by epithelium
#include <curand_kernel.h>
#include <thrust/fill.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/property.cuh"
#include "../lib/protrusions.cuh"
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
Property<N_MAX, CELL_TYPES> type;
Property<N_MAX, int> n_mes_nbs;
Property<N_MAX, int> n_epi_nbs;
curandState *d_state;


__device__ CELL_TYPES* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;

__device__ pocell relu_w_polarity(pocell Xi, pocell Xj, int i, int j) {
    pocell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0)/2;
    } else {
        F = fmaxf(0.8 - dist, 0)*2 - fmaxf(dist - 0.9, 0)/2;
    }
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if (d_type[j] == MESENCHYME) d_mes_nbs[i] += 1;
    else d_epi_nbs[i] += 1;

    if (d_type[i] == MESENCHYME or d_type[j] == MESENCHYME) return dF;

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ auto d_relu_w_polarity = &relu_w_polarity;
auto h_relu_w_polarity = get_device_object(d_relu_w_polarity, 0);


__global__ void proliferate(float rate, float mean_distance, pocell* d_X, int* d_n_cells,
        curandState* d_state) {
    D_ASSERT(*d_n_cells*rate <= N_MAX);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *d_n_cells*(1 - rate)) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case MESENCHYME: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
        }
        case EPITHELIUM: {
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].phi = d_X[i].phi;
    d_X[n].theta = d_X[i].theta;
    d_type[n] = d_type[i] == MESENCHYME ? MESENCHYME : EPITHELIUM;
    d_mes_nbs[n] = 0;
    d_epi_nbs[n] = 0;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    bolls.set_n(200);
    uniform_sphere(MEAN_DIST, bolls);
    for (auto i = 0; i < bolls.get_n(); i++) type.h_prop[i] = MESENCHYME;
    type.memcpyHostToDevice();
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    cudaMalloc(&d_state, N_MAX*sizeof(curandState));
    setup_rand_states<<<(N_MAX + 128 - 1)/128, 128>>>(d_state, N_MAX);

    // Relax
    for (auto time_step = 0; time_step <= 500; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_n(), 0);
        bolls.step(DELTA_T, h_relu_w_polarity);
    }

    // Find epithelium
    bolls.memcpyDeviceToHost();
    n_mes_nbs.memcpyDeviceToHost();
    for (auto i = 0; i < bolls.get_n(); i++) {
        if (n_mes_nbs.h_prop[i] < 12*2) {  // 2nd order solver
            type.h_prop[i] = EPITHELIUM;
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
    type.memcpyHostToDevice();

    // Simulate growth
    VtkOutput sim_output("passive_growth");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        type.memcpyDeviceToHost();
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_n(), 0);
        bolls.step(DELTA_T, h_relu_w_polarity);
        proliferate<<<(bolls.get_n() + 128 - 1)/128, 128>>>(RATE*(time_step > 100),
            MEAN_DIST, bolls.d_X, bolls.d_n, d_state);
        sim_output.write_positions(bolls);
        sim_output.write_property(type);
        sim_output.write_polarity(bolls);
    }

    return 0;
}
