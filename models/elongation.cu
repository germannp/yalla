// Simulate elongation of semisphere
#include <stdio.h>
#include <thread>
#include <functional>
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../lib/cudebug.cuh"
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/links.cuh"
#include "../lib/polarity.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto n_0 = 5000;
const auto n_max = 61000;
const auto r_max = 1;
const auto r_protrusion = 1.5;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.5;
const auto n_time_steps = 500;
const auto skip_steps = 5;
const auto dt = 0.2;
enum Cell_types {mesenchyme, epithelium, aer};

MAKE_PT(Lb_cell, x, y, z, w, f, phi, theta);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ Lb_cell pairwise_interaction(Lb_cell Xi, Lb_cell Xj, int i, int j) {
    Lb_cell dF {0};
    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = (d_type[i] > mesenchyme) - 0.01*Xi.w;
        dF.f = (d_type[i] == aer) - 0.01*Xi.f;
        return dF;
    }

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
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
    auto D = dist < r_max ? 0.1 : 0;
    dF.w = - r.w*D;
    dF.f = - r.f*D;

    if (d_type[j] == mesenchyme) d_mes_nbs[i] += 1;
    else d_epi_nbs[i] += 1;

    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += rigidity_force(Xi, Xj)*0.2;
    return dF;
}

#include "../lib/solvers.cuh"


__global__ void update_protrusions(const Lattice<n_max>* __restrict__ d_lattice,
        const Lb_cell* __restrict d_X, int n_cells, Link* d_link, curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5)/prots_per_cell);
    auto rand_nb_cube = d_lattice->d_cube_id[j]
        + d_moore_nhood[min(static_cast<int>(curand_uniform(&d_state[i])*27), 26)];
    auto cells_in_cube = d_lattice->d_cube_end[rand_nb_cube] - d_lattice->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto cell_a = d_lattice->d_cell_id[j];
    auto cell_b = d_lattice->d_cell_id[d_lattice->d_cube_start[rand_nb_cube]
        + min(static_cast<int>(curand_uniform(&d_state[i])*cells_in_cube), cells_in_cube - 1)];
    D_ASSERT(cell_a >= 0); D_ASSERT(cell_a < n_cells);
    D_ASSERT(cell_b >= 0); D_ASSERT(cell_b < n_cells);
    if (cell_a == cell_b) return;

    auto r = d_X[cell_a] - d_X[cell_b];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    auto both_mesenchyme = (d_type[cell_a] == mesenchyme)
        and (d_type[cell_b] == mesenchyme);
    auto along_w = fabs(r.w/(d_X[cell_a].w + d_X[cell_b].w)) > 0.2;
    auto high_f = (d_X[cell_a].f + d_X[cell_b].f) > 0.2;
    if (both_mesenchyme and (dist < r_protrusion) and (along_w or high_f)) {
        d_link[cell_a*prots_per_cell + i%prots_per_cell].a = cell_a;
        d_link[cell_a*prots_per_cell + i%prots_per_cell].b = cell_b;
    }
}


__global__ void proliferate(float rate, float mean_distance, Lb_cell* d_X,
        int* d_n_cells, curandState* d_state) {
    D_ASSERT(*d_n_cells*rate <= n_max);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *d_n_cells*(1 - rate)) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
            break;
        }
        default:
            if (d_epi_nbs[i] > d_mes_nbs[i]) return;
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].w = d_X[i].w/2;
    d_X[i].w = d_X[i].w/2;
    d_X[n].phi = d_X[i].phi;
    d_X[n].theta = d_X[i].theta;
    d_type[n] = d_type[i];
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Lb_cell, n_max, Lattice_solver> bolls(n_0);
    uniform_sphere(0.733333, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        bolls.h_X[i].x = fabs(bolls.h_X[i].x);
        bolls.h_X[i].y = bolls.h_X[i].y/1.5;
        bolls.h_X[i].w = 0;
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    Links<static_cast<int>(n_max*prots_per_cell)> protrusions(protrusion_strength,
        n_0*prots_per_cell);
    auto intercalation = std::bind(
        linear_force<static_cast<int>(n_max*prots_per_cell), Lb_cell>,
        protrusions, std::placeholders::_1, std::placeholders::_2);
    for (auto i = 0; i < n_max*prots_per_cell; i++) {
        protrusions.h_link[i].a = static_cast<int>((i + 0.5)/prots_per_cell);
        protrusions.h_link[i].b = static_cast<int>((i + 0.5)/prots_per_cell);
    }
    protrusions.copy_to_device();

    // Relax
    for (auto time_step = 0; time_step <= 200; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step(dt);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 12*2 and bolls.h_X[i].x > 0) {  // 2nd order solver
            if (fabs(bolls.h_X[i].y) < 0.75 and bolls.h_X[i].x > 3)
                type.h_prop[i] = aer;
            else
                type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
                + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
        } else {
            bolls.h_X[i].phi = 0;
            bolls.h_X[i].theta = 0;
        }
        bolls.h_X[i].w = 0;
        bolls.h_X[i].f = 0;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    bolls.take_step(dt);  // Relax epithelium before proliferate

    // Simulate diffusion & intercalation
    Vtk_output output("elongation");
    for (auto time_step = 0; time_step <= n_time_steps/skip_steps; time_step++) {
        bolls.copy_to_host();
        protrusions.copy_to_host();
        type.copy_to_host();

        std::thread calculation([&] {
            for (auto i = 0; i < skip_steps; i++) {
                proliferate<<<(bolls.get_d_n() + 128 - 1)/128, 128>>>(0.005, 0.733333, bolls.d_X,
                    bolls.d_n, protrusions.d_state);
                protrusions.set_d_n(bolls.get_d_n()*prots_per_cell);
                bolls.build_lattice(r_protrusion);
                update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
                    bolls.d_X, bolls.get_d_n(), protrusions.d_link, protrusions.d_state);
                thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
                thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);
                bolls.take_step(dt, intercalation);
            }
        });

        output.write_positions(bolls);
        output.write_links(protrusions);
        output.write_property(type);
        // output.write_polarity(bolls);
        output.write_field(bolls, "Wnt");
        output.write_field(bolls, "Fgf", &Lb_cell::f);

        calculation.join();
    }

    return 0;
}
