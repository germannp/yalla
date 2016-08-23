// Simulate elongation of semisphere
#include <thread>
#include <functional>
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/protrusions.cuh"
#include "../lib/epithelium.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto N_CELLS_0 = 5000;
const auto N_MAX = 61000;
const auto R_MAX = 1;
const auto R_LINK = 1.5;
const auto LINKS_P_CELL = 1.f;  // Must be >= 1 as rand states used to proliferate
const auto LINK_STRENGTH = 0.5;
const auto N_TIME_STEPS = 500;
const auto SKIP_STEPS = 5;
const auto DELTA_T = 0.2;
enum CELL_TYPES {MESENCHYME, EPITHELIUM};

MAKE_PT(lbcell, x, y, z, w, phi, theta);


__device__ CELL_TYPES* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

__device__ lbcell pairwise_interaction(lbcell Xi, lbcell Xj, int i, int j) {
    lbcell dF {0};
    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = (d_type[i] > MESENCHYME) - 0.01*Xi.w;
        return dF;
    }

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
    auto D = dist < R_MAX ? 0.1 : 0;
    dF.w = - r.w*D;

    if (d_type[j] == MESENCHYME) d_mes_nbs[i] += 1;
    else d_epi_nbs[i] += 1;

    if (d_type[i] == MESENCHYME or d_type[j] == MESENCHYME) return dF;

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

#include "../lib/solvers.cuh"


__global__ void update_links(const Lattice<N_MAX>* __restrict__ d_lattice,
        const lbcell* __restrict d_X, int n_cells, Link* d_link, curandState* d_state) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_cells*LINKS_P_CELL) return;

    auto j = static_cast<int>(curand_uniform(&d_state[i])*n_cells);
    auto rand_cube = d_lattice->d_cube_id[j]
        +  static_cast<int>(curand_uniform(&d_state[i])*3) - 1
        + (static_cast<int>(curand_uniform(&d_state[i])*3) - 1)*LATTICE_SIZE
        + (static_cast<int>(curand_uniform(&d_state[i])*3) - 1)*LATTICE_SIZE*LATTICE_SIZE;
    auto cells_in_cube = d_lattice->d_cube_end[rand_cube] - d_lattice->d_cube_start[rand_cube];
    if (cells_in_cube < 1) return;

    auto k = d_lattice->d_cube_start[rand_cube]
        + static_cast<int>(curand_uniform(&d_state[i])*cells_in_cube);
    auto r = d_X[d_lattice->d_cell_id[j]] - d_X[d_lattice->d_cell_id[k]];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if ((j != k) and (d_type[d_lattice->d_cell_id[j]] == MESENCHYME)
            and (d_type[d_lattice->d_cell_id[k]] == MESENCHYME)
            and (dist < R_LINK)
            and (fabs(r.w/(d_X[d_lattice->d_cell_id[j]].w + d_X[d_lattice->d_cell_id[k]].w)) > 0.2)) {
            // and (fabs(r.x/dist) < 0.2) and (j != k) and (dist < 2)) {
        d_link[i].a = d_lattice->d_cell_id[j];
        d_link[i].b = d_lattice->d_cell_id[k];
    }
}


__global__ void proliferate(float rate, float mean_distance, lbcell* d_X, int* d_n_cells,
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
    d_X[n].w = d_X[i].w/2;
    d_X[i].w = d_X[i].w/2;
    d_X[n].phi = d_X[i].phi;
    d_X[n].theta = d_X[i].theta;
    d_type[n] = d_type[i];
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<lbcell, N_MAX, LatticeSolver> bolls;
    bolls.set_n(N_CELLS_0);
    uniform_sphere(0.733333, bolls);
    Property<N_MAX, CELL_TYPES> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < bolls.get_n(); i++) {
        bolls.h_X[i].x = fabs(bolls.h_X[i].x);
        bolls.h_X[i].y = bolls.h_X[i].y/1.5;
        bolls.h_X[i].w = 0;
        type.h_prop[i] = MESENCHYME;
    }
    bolls.memcpyHostToDevice();
    type.memcpyHostToDevice();
    Property<N_MAX, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<N_MAX, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    Protrusions<static_cast<int>(N_MAX*LINKS_P_CELL)> links(LINK_STRENGTH);
    auto intercalation = std::bind(
        link_forces<static_cast<int>(N_MAX*LINKS_P_CELL), lbcell>,
        links, std::placeholders::_1, std::placeholders::_2);

    // Relax
    for (auto time_step = 0; time_step <= 200; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_n(), 0);
        bolls.step(DELTA_T);
    }

    // Find epithelium
    bolls.memcpyDeviceToHost();
    n_mes_nbs.memcpyDeviceToHost();
    for (auto i = 0; i < bolls.get_n(); i++) {
        if (n_mes_nbs.h_prop[i] < 12*2 and bolls.h_X[i].x > 0) {  // 2nd order solver
            type.h_prop[i] = EPITHELIUM;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
                + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
        } else {
            bolls.h_X[i].phi = 0;
            bolls.h_X[i].theta = 0;
        }
        bolls.h_X[i].w = 0;
    }
    bolls.memcpyHostToDevice();
    type.memcpyHostToDevice();
    bolls.step(DELTA_T);  // Relax epithelium before proliferate

    // Simulate diffusion & intercalation
    VtkOutput sim_output("elongation");
    for (auto time_step = 0; time_step <= N_TIME_STEPS/SKIP_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        links.memcpyDeviceToHost();
        links.set_n(bolls.get_n()*LINKS_P_CELL);
        type.memcpyDeviceToHost();

        std::thread calculation([&] {
            for (auto i = 0; i < SKIP_STEPS; i++) {
                proliferate<<<(bolls.get_n() + 128 - 1)/128, 128>>>(0.005, 0.733333, bolls.d_X,
                    bolls.d_n, links.d_state);
                bolls.build_lattice(R_LINK);
                update_links<<<(links.get_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
                    bolls.d_X, bolls.get_n(), links.d_link, links.d_state);
                thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_n(), 0);
                thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_n(), 0);
                bolls.step(DELTA_T, intercalation);
            }
        });

        sim_output.write_positions(bolls);
        sim_output.write_protrusions(links);
        sim_output.write_property(type);
        // sim_output.write_polarity(bolls);
        sim_output.write_field(bolls, "Wnt");

        calculation.join();
    }

    return 0;
}
