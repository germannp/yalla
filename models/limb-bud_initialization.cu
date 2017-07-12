// Proliferating mesenchyme between two epithelial layers
#include <stdio.h>
#include <thread>
#include <functional>
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../include/cudebug.cuh"
#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/vtk.cuh"


const auto n_0 = 3000;
const auto n_max = 61000;
const auto proliferation_rate = 0.01;
const auto r_max = 1.f;
const auto mean_distance = 0.75;
const auto prots_per_cell = 1;
const auto n_time_steps = 500;
const auto dt = 0.2;
enum Cell_types {mesoderm, mesenchyme, ectoderm, aer};

MAKE_PT(Lb_cell, w, f, theta, phi);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Lb_cell lb_force(Lb_cell Xi, Lb_cell r, float dist, int i, int j) {
    Lb_cell dF {0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0)*2 - fmaxf(dist - 0.8, 0);
    } else {
        F = fmaxf(0.8 - dist, 0)*2 - fmaxf(dist - 0.9, 0);
    }
    dF.x = r.x*F/dist*(d_type[i] != mesoderm);
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if (d_type[j] == mesenchyme) { d_mes_nbs[i] += 1; return dF; }
    else d_epi_nbs[i] += 1;

    if (d_type[i] != d_type[j]) return dF;

    dF += rigidity_force(Xi, r, dist)*0.1;
    return dF;
}


__global__ void proliferate(int n_0, Lb_cell* d_X, int* d_n_cells, curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_0) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > proliferation_rate) return;
            break;
        }
        default:
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
    }

    auto n = atomicAdd(d_n_cells, 1);
    D_ASSERT(n <= n_max);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].w = d_X[i].w/2;
    d_X[i].w = d_X[i].w/2;
    d_X[n].f = d_X[i].f/2;
    d_X[i].f = d_X[i].f/2;
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Lb_cell, n_max, Grid_solver> bolls(n_0);
    uniform_circle(r_max/2, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0/2; i++) {
        bolls.h_X[i].y /= 1.5;
        bolls.h_X[i].theta = -M_PI/2;
        type.h_prop[i] = mesoderm;
        bolls.h_X[i + n_0/2].x += mean_distance/2;
        bolls.h_X[i + n_0/2].y /= 1.5;
        bolls.h_X[i + n_0/2].theta = M_PI/2;
        type.h_prop[i + n_0/2] = ectoderm;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    for (auto i = 0; i < 100; i++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<lb_force>(dt);
    }
    bolls.copy_to_host();
    *bolls.h_n += 100;
    uniform_circle(mean_distance*1.5, bolls, n_0);
    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = 0.1;
        bolls.h_X[i].y /= 1.5;
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Links<n_max*prots_per_cell> protrusions(0.2f, n_0*prots_per_cell);

    // Proliferate
    Vtk_output output("initialization");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        type.copy_to_host();
        if (time_step < 400) {
            proliferate<<<(bolls.get_d_n() + 128 - 1)/128, 128>>>(bolls.get_d_n(), bolls.d_X, bolls.d_n,
                protrusions.d_state);
        }
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<lb_force>(dt);
        output.write_positions(bolls);
        output.write_property(type);
        output.write_polarity(bolls);
    }

    return 0;
}
