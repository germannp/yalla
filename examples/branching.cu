// This program simulates a branching mechanism on a spheric organoid.
// A Turing mechanism taking place on its surface creates a pattern,
// peaks of activator induce local proliferation on the underlying cells,
// resulting on the growth of a branch.
#include <string>
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/links.cuh"


const auto r_max_epi = 1.5;
const auto r_max_mes = 1.0;
const auto r_min_homotypic = 0.6;
const auto r_min_heterotypic = 0.8;
const auto skip_steps = 100;
const auto lambda = 1.;

// Turing parameters
const auto D_u = 0.05;
const auto D_v = 2.0;
const auto f_v = 1.0;
const auto f_u = 80.0;
const auto g_u = 80.0;
const auto m_u = 0.25;  // degradation rates
const auto m_v = 0.75;
const auto s_u = 0.05;

const auto r_min_min = min(r_min_homotypic,r_min_heterotypic);
const auto r_max_max = max(r_max_epi,r_max_mes);
const auto dt = 0.05*r_min_min*r_min_min/D_v;

const auto epi_proliferation_rate = 0.000893;
const auto mes_proliferation_rate = 0.000493;

// threshold conc. of v that allows mesench. cells to divide
const auto prolif_threshold = 1600.0f;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

MAKE_PT(Cell, theta, phi, u, v);


__device__ Cell epi_turing_mes_noturing(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    // Meinhard equations
    if (i == j) {
        if (d_type[i] == epithelium) {
            dF.u = lambda*((f_u*Xi.u*Xi.u)/(1 + f_v*Xi.v) - m_u*Xi.u + s_u);
            dF.v = lambda*(g_u*Xi.u*Xi.u - m_v*Xi.v);

            // prevent negative values
            if(-dF.u > Xi.u) dF.u = 0.0f;
            if(-dF.v > Xi.v) dF.v = 0.0f;
        }
        return dF;
    }

    float r_max;
    float r_min;
    if (d_type[i] == d_type[j]) {
        r_min=r_min_homotypic;
        if (d_type[i]==epithelium) r_max = r_max_epi;
        else r_max = r_max_mes;
    } else {
        r_min = r_min_heterotypic;
        r_max = r_max_mes;
    }

    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    // Diffusion
    if(d_type[i] == epithelium && d_type[j] == epithelium) {
        dF.u = -D_u*r.u;
        dF.v = -D_v*r.v;

        if(-dF.u > Xi.u) dF.u=0.0f;
        if(-dF.v > Xi.v) dF.v=0.0f;

        dF += rigidity_force(Xi, r, dist)*0.2;
    } else {
        dF.v = -D_v*r.v;  // inhibitor diffuses towards the mesenchyme to induce proliferation
    }

    if (d_type[j] == epithelium) atomicAdd(&d_epi_nbs[i], 1);
    else atomicAdd(&d_mes_nbs[i],1);

    return dF;
}


__global__ void proliferate(float mean_distance, Cell* d_X, int* d_n_cells,
        curandState* d_state) {
    D_ASSERT(*d_n_cells*epi_proliferation_rate <= n_max);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *d_n_cells*(1 - epi_proliferation_rate)) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            if(d_X[i].v < prolif_threshold) return;
            auto r = curand_uniform(&d_state[i]);
            if (r > mes_proliferation_rate) return;
            break;
        }
        case epithelium: {
            if (d_epi_nbs[i] > 20) return;

            auto r = curand_uniform(&d_state[i]);
            if (r > epi_proliferation_rate) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].u = d_X[i].u/2;
    d_X[i].u = d_X[i].u/2;
    d_X[n].v = d_X[i].v/2;
    d_X[i].v = d_X[i].v/2;
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
}


int main(int argc, char const *argv[]) {
    // Command line arguments
    // argv[1] : output file taking
    // argv[2] : number of time steps
    auto n_time_steps = std::stoi(argv[2]);

    // Initial state
    Solution<Cell, n_max, Grid_solver> bolls(n_0);
    uniform_sphere(0.5, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        bolls.h_X[i].u = 0;
        bolls.h_X[i].v = 0;
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // State for proliferations
    curandState *d_state;
    cudaMalloc(&d_state, n_max*sizeof(curandState));
    setup_rand_states<<<(n_max + 128 - 1)/128, 128>>>(d_state, n_max);

    // Relax
    for (auto time_step = 0; time_step <= 500; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<epi_turing_mes_noturing>(dt/100);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 35) {
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
                + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);

            bolls.h_X[i].u = rand()/(RAND_MAX + 1.)/5 - 0.1;
            bolls.h_X[i].v = rand()/(RAND_MAX + 1.)/5 - 0.1;

        } else {
            bolls.h_X[i].theta = 0;
            bolls.h_X[i].phi = 0;
        }

    }
    bolls.copy_to_device();
    type.copy_to_device();

    // Relax again to let epithelium stabilise
    for (auto time_step = 0; time_step <= 100; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<epi_turing_mes_noturing>(dt);
    }

    // Integrate positions
    Vtk_output output(argv[1]);
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();

        proliferate<<<(bolls.get_d_n() + 128 - 1)/128, 128>>>(0.75, bolls.d_X,
            bolls.d_n, d_state);
        bolls.build_grid(r_max_max);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);

        bolls.take_step<epi_turing_mes_noturing>(dt);
        if(time_step%skip_steps == 0) {
            output.write_positions(bolls);
            output.write_polarity(bolls);
            output.write_field(bolls, "u", &Cell::u);
            output.write_field(bolls, "v", &Cell::v);
        }
    }

    return 0;
}
