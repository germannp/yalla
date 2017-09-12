// Simulates branching on a spheroid induced by Turing mechanism on surface
#include <curand_kernel.h>
#include <time.h>
#include <string>
#include <thread>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

const auto r_max = 1.0f;
const auto n_time_steps = 100;
const auto skip_steps = 10;
const auto lambda = 0.01;

// Turing parameters
const auto D_u = 0.005;
const auto D_v = 0.2;
const auto f_v = 1.0;
const auto f_u = 80.0;
const auto g_u = 80.0;
const auto m_u = 0.25;  // degradation rates
const auto m_v = 0.75;
const auto s_u = 0.05;

const auto dt = 0.2f;

const auto epi_proliferation_rate = 0.2;
const auto mes_proliferation_rate = 0.1;

// Threshold conc. of v that allows mesench. cells to divide
const auto prolif_threshold = 160.0f;

const auto n_0 = 500;
const auto n_max = 65000;

enum Cell_types { mesenchyme, epithelium };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

MAKE_PT(Cell, theta, phi, u, v);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};
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

    if (d_type[i] == epithelium && d_type[j] == epithelium)
        dF += rigidity_force(Xi, r, dist) * 0.2;

    if (d_type[j] == epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    return dF;
}


__device__ Cell epi_turing_mes_noturing(
    Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    // Meinhard equations
    if (i == j) {
        if (d_type[i] == epithelium) {
            dF.u = lambda *
                   ((f_u * Xi.u * Xi.u) / (1 + f_v * Xi.v) - m_u * Xi.u + s_u);
            dF.v = lambda * (g_u * Xi.u * Xi.u - m_v * Xi.v);

            // Prevent negative values
            if (-dF.u > Xi.u) dF.u = 0.0f;
            if (-dF.v > Xi.v) dF.v = 0.0f;
        }
        return dF;
    }

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

    // Diffusion
    if (d_type[i] == epithelium && d_type[j] == epithelium) {
        dF.u = -D_u * r.u;
        dF.v = -D_v * r.v;

        if (-dF.u > Xi.u) dF.u = 0.0f;
        if (-dF.v > Xi.v) dF.v = 0.0f;

        dF += rigidity_force(Xi, r, dist) * 0.2;
    } else {
        dF.v = -D_v * r.v;  // Diffuses into mesenchyme to induce proliferation
    }

    if (d_type[j] == epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    return dF;
}


__global__ void proliferate(
    float mean_distance, Cell* d_X, int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * epi_proliferation_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - epi_proliferation_rate))
        return;  // Dividing new cells is problematic!

    auto rnd = curand_uniform(&d_state[i]);
    switch (d_type[i]) {
        case mesenchyme: {
            if (d_X[i].v < prolif_threshold) return;

            if (rnd > mes_proliferation_rate) return;

            break;
        }
        case epithelium: {
            if (d_epi_nbs[i] > 10) return;

            if (d_mes_nbs[i] <= 0) return;

            if (rnd > epi_proliferation_rate) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].u = d_X[i].u / 2;
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    d_X[i].u = d_X[i].u / 2;
    d_X[n].v = d_X[i].v / 2;
    d_X[i].v = d_X[i].v / 2;
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}


int main(int argc, char const* argv[])
{
    // Initial state
    Solution<Cell, n_max, Grid_solver> bolls(n_0);
    random_sphere(0.75, bolls);
    Property<n_max, Cell_types> type("type");
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs("n_mes_nbs");
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs("n_epi_nbs");
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // State for proliferations
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    // Relax
    for (auto time_step = 0; time_step <= 500; time_step++) {
        thrust::fill(
            thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<relaxation_force, friction_on_background>(dt);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 24) {
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x * bolls.h_X[i].x +
                              bolls.h_X[i].y * bolls.h_X[i].y +
                              bolls.h_X[i].z * bolls.h_X[i].z);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z / dist);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);

            bolls.h_X[i].u = rand() / (RAND_MAX + 1.) / 5 - 0.1;
            bolls.h_X[i].v = rand() / (RAND_MAX + 1.) / 5 - 0.1;
        }
    }
    bolls.copy_to_device();
    type.copy_to_device();

    // Relax again to let epithelium stabilise
    for (auto time_step = 0; time_step <= 500; time_step++) {
        proliferate<<<(bolls.get_d_n() + 128 - 1) / 128, 128>>>(
            0.75, bolls.d_X, bolls.d_n, d_state);
        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + bolls.get_d_n(), 0);
        bolls.take_step<relaxation_force>(dt);
    }

    // Integrate positions
    Vtk_output output("branching");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        type.copy_to_host();
        // n_epi_nbs.copy_to_host();

        std::thread calculation([&] {
            for (auto i = 0; i <= skip_steps; i++) {
                proliferate<<<(bolls.get_d_n() + 128 - 1) / 128, 128>>>(
                    0.75, bolls.d_X, bolls.d_n, d_state);
                thrust::fill(thrust::device, n_mes_nbs.d_prop,
                    n_mes_nbs.d_prop + bolls.get_d_n(), 0);
                thrust::fill(thrust::device, n_epi_nbs.d_prop,
                    n_epi_nbs.d_prop + bolls.get_d_n(), 0);
                bolls.take_step<epi_turing_mes_noturing>(dt);
            }
        });

        output.write_positions(bolls);
        output.write_polarity(bolls);
        output.write_field(bolls, "u", &Cell::u);
        output.write_field(bolls, "v", &Cell::v);
        output.write_property(type);
        // output.write_property(n_epi_nbs);

        calculation.join();
    }

    return 0;
}
