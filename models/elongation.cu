// Simulate elongation of semisphere
#include <thread>
#include <functional>
#include <curand_kernel.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/protrusions.cuh"
#include "../lib/epithelium.cuh"
#include "../lib/property.cuh"
#include "../lib/vtk.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_MAX = 61000;
const auto R_LINK = 1.5;
const auto LINKS_P_CELL = 1.f;  // Must be >= 1 as rand states used to proliferatie
const auto LINK_STRENGTH = 0.5;
const auto N_TIME_STEPS = 500;
const auto DELTA_T = 0.2;
enum CELL_TYPES {MESENCHYME, STRETCHED_EPI, EPITHELIUM};

MAKE_PT(lbcell, x, y, z, w, phi, theta);


__device__ CELL_TYPES* d_type;

__device__ lbcell cubic_w_diffusion(lbcell Xi, lbcell Xj, int i, int j) {
    lbcell dF {0};
    if (i == j) {
        D_ASSERT(Xi.w >= 0);
        dF.w = (d_type[i] > MESENCHYME) - 0.01*Xi.w;
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

    if (d_type[i] == MESENCHYME or d_type[j] == MESENCHYME) return dF;

    if (dist < 0.733333) d_type[i] = EPITHELIUM;
    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ auto d_cubic_w_diffusion = &cubic_w_diffusion;
auto h_cubic_w_diffusion = get_device_object(d_cubic_w_diffusion, 0);


__device__ lbcell count_neighbours(lbcell Xi, lbcell Xj, int i, int j) {
    lbcell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF.w = dist < R_MAX ? 1 : 0;
    return dF;
}

__device__ auto d_count_neighbours = &count_neighbours;
auto h_count_neighbours = get_device_object(d_count_neighbours, 0);


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

    if (d_type[i] == EPITHELIUM) {
        d_type[i] = STRETCHED_EPI;
        return;
    }

    if (d_type[i] == MESENCHYME) {
        auto r = curand_uniform(&d_state[i]);
        if (r > rate) return;
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
    d_type[n] = d_type[i] == MESENCHYME ? MESENCHYME : STRETCHED_EPI;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<lbcell, N_MAX, LatticeSolver> bolls;
    bolls.set_n(5000);
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
    Protrusions<static_cast<int>(N_MAX*LINKS_P_CELL)> links(LINK_STRENGTH);
    auto intercalation = std::bind(
        link_forces<static_cast<int>(N_MAX*LINKS_P_CELL), lbcell>,
        links, std::placeholders::_1, std::placeholders::_2);

    // Relax
    VtkOutput relax_output("relaxation");
    for (auto time_step = 0; time_step <= 200; time_step++) {
        bolls.step(DELTA_T, h_cubic_w_diffusion);
        relax_output.print_progress();
    }
    relax_output.print_done();

    // Find epithelium
    bolls.step(1, h_count_neighbours);
    bolls.memcpyDeviceToHost();
    for (auto i = 0; i < bolls.get_n(); i++) {
        if (bolls.h_X[i].w < 12 and bolls.h_X[i].x > 0) {
            type.h_prop[i] = STRETCHED_EPI;
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
    bolls.step(DELTA_T, h_cubic_w_diffusion);  // Relax epithelium before proliferate

    // Simulate diffusion & intercalation
    VtkOutput sim_output("elongation");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        links.memcpyDeviceToHost();
        links.set_n(bolls.get_n()*LINKS_P_CELL);
        type.memcpyDeviceToHost();

        std::thread calculation([&] {
            proliferate<<<(bolls.get_n() + 128 - 1)/128, 128>>>(0.005, 0.733333, bolls.d_X,
                bolls.d_n, links.d_state);
            bolls.build_lattice(R_LINK);
            update_links<<<(links.get_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
                bolls.d_X, bolls.get_n(), links.d_link, links.d_state);
            bolls.step(DELTA_T, h_cubic_w_diffusion, intercalation);
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
