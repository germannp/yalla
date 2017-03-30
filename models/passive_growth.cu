// Simulate growing mesenchyme envelopped by epithelium
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/inits.cuh"
#include "../lib/property.cuh"
#include "../lib/links.cuh"
#include "../lib/vtk.cuh"
#include "../lib/polarity.cuh"


const auto r_max = 1;
const auto mean_dist = 0.75;
const auto prolif_rate = 0.006;
const auto n_0 = 200;
const auto n_max = 5000;
const auto n_time_steps = 500;
const auto dt = 0.2;
enum Cell_types {mesenchyme, epithelium};


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Po_cell relu_w_epithelium(Po_cell Xi, Po_cell Xj, int i, int j) {
    Po_cell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0)/2;
    } else {
        F = fmaxf(0.8 - dist, 0)*2 - fmaxf(dist - 0.9, 0)/2;
    }
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if (d_type[j] == mesenchyme) d_mes_nbs[i] += 1;
    else d_epi_nbs[i] += 1;

    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += rigidity_force(Xi, Xj)*0.2;
    return dF;
}


__global__ void proliferate(float rate, float mean_distance, Po_cell* d_X, int* d_n_cells,
        curandState* d_state) {
    D_ASSERT(*d_n_cells*rate <= n_max);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *d_n_cells*(1 - rate)) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
        }
        case epithelium: {
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_mes_nbs[n] = 0;
    d_epi_nbs[n] = 0;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Po_cell, n_max, Lattice_solver> bolls(n_0);
    uniform_sphere(mean_dist, bolls);
    Property<n_max, Cell_types> type;
    for (auto i = 0; i < n_0; i++) type.h_prop[i] = mesenchyme;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    curandState *d_state;
    cudaMalloc(&d_state, n_max*sizeof(curandState));
    setup_rand_states<<<(n_max + 128 - 1)/128, 128>>>(d_state, n_max);

    // Relax
    for (auto time_step = 0; time_step <= 500; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<relu_w_epithelium>(dt);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 12*2) {  // 2nd order solver
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x + bolls.h_X[i].y*bolls.h_X[i].y
                + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);
        } else {
            bolls.h_X[i].theta = 0;
            bolls.h_X[i].phi = 0;
        }
    }
    bolls.copy_to_device();
    type.copy_to_device();

    // Simulate growth
    Vtk_output sim_output("passive_growth");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        type.copy_to_host();
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<relu_w_epithelium>(dt);
        proliferate<<<(bolls.get_d_n() + 128 - 1)/128, 128>>>(prolif_rate*(time_step > 100),
            mean_dist, bolls.d_X, bolls.d_n, d_state);
        sim_output.write_positions(bolls);
        sim_output.write_property(type);
        sim_output.write_polarity(bolls);
    }

    return 0;
}
