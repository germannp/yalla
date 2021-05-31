// Simulate growing mesenchyme envelopped by epithelium
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <time.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto mean_dist = 0.75;
const auto prolif_rate = 0.006;
const auto n_0 = 200;
const auto n_max = 5000;
const auto n_time_steps = 500;
const auto dt = 0.2;
enum Cell_types { mesenchyme, epithelium };


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Po_cell relu_w_epithelium(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    } else {
        F = fmaxf(0.8 - dist, 0) * 2 - fmaxf(dist - 0.9, 0);
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    if (d_type[j] == mesenchyme)
        d_mes_nbs[i] += 1;
    else
        d_epi_nbs[i] += 1;

    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += bending_force(Xi, r, dist) * 0.15;
    return dF;
}


__global__ void proliferate(float rate, int n_cells, curandState* d_state,
    Po_cell* d_X, float3* d_old_v, int* d_n_cells)
{
    D_ASSERT(n_cells * rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto rnd = curand_uniform(&d_state[i]);
            if (rnd > rate) return;

            break;
        }
        case epithelium: {
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
    d_X[n].x = d_X[i].x + mean_dist / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_dist / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_dist / 4 * cosf(theta);
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_mes_nbs[n] = 0;
    d_epi_nbs[n] = 0;
    d_old_v[n] = d_old_v[i];
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Grid_solver> cells{n_max};
    *cells.h_n = n_0;
    relaxed_sphere(mean_dist, cells);
    Property<Cell_types> type{n_max};
    for (auto i = 0; i < n_0; i++) type.h_prop[i] = mesenchyme;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    type.copy_to_device();
    Property<int> n_mes_nbs{n_max};
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<int> n_epi_nbs{n_max};
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    auto reset_nbs = [&](int n, const Po_cell* __restrict__ d_X, Po_cell* d_dX) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);
    };

    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    // Find epithelium
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
    cells.take_step<relu_w_epithelium>(dt);
    cells.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 12 * 2) {  // *2 for 2nd order solver
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                              cells.h_X[i].y * cells.h_X[i].y +
                              cells.h_X[i].z * cells.h_X[i].z);
            cells.h_X[i].theta = acosf(cells.h_X[i].z / dist);
            cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x);
        } else {
            cells.h_X[i].theta = 0;
            cells.h_X[i].phi = 0;
        }
    }
    cells.copy_to_device();
    type.copy_to_device();

    // Simulate growth
    Vtk_output output{"passive_growth"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        type.copy_to_host();
        cells.take_step<relu_w_epithelium>(dt, reset_nbs);
        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            prolif_rate * (time_step > 100), cells.get_d_n(), d_state,
            cells.d_X, cells.d_old_v, cells.d_n);
        output.write_positions(cells);
        output.write_property(type);
        output.write_polarity(cells);
    }

    return 0;
}
