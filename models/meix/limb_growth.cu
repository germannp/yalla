// Simulation of limb bud growth starting with realistic limb bud shape

// Command line arguments
// argv[1]=input file tag
// argv[2]=output file tag
// argv[3]=proliferation rate
// argv[4]=time steps
// argv[4]=proliferation rate distr. (0=uniform, 1=PD gradient)

#include <curand_kernel.h>
#include <time.h>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/links.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1f;
const auto n_max = 150000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;

enum Cell_types { mesenchyme, epithelium, aer };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
__device__ float* d_prolif_rate;

Property<n_max, int> n_mes_nbs("n_mes_nbs");  // defining these here so function
Property<n_max, int> n_epi_nbs("n_epi_nbs");  // "neighbour_init" can see them

MAKE_PT(Cell, w, f, theta, phi);

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        if (Xi.w < 0.f) dF.w = 0.f;
        if (Xi.f < 0.f) dF.f = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else if (d_type[i] > mesenchyme && d_type[j] > mesenchyme) {
        F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 3.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.5f;
    dF.f = -r.f * (d_type[i] == mesenchyme) * 0.5f;

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += rigidity_force(Xi, r, dist) * 0.1f;


    if (d_type[j] >= epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    // if (Xi.x < 1.f) {
    //     dF.x = 0.f;
    //     dF.f = 0.f;
    //     dF.w = 0.f;
    // }

    if (Xi.w<0.f) dF.w=0.f;
    if (Xi.f<0.f) dF.f=0.f;
    return dF;
}

__device__ float wall_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (i == j) return 0;
    // if (Xi.x < 1.0f) return 0;
    return 1;
}

__device__ void link_force(const Cell* __restrict__ d_X, const int a,
    const int b, const float strength, Cell* d_dX)
{
    if(d_X[a].f + d_X[b].f> 0.2f) return;

    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -strength * r.x / dist);
    atomicAdd(&d_dX[a].y, -strength * r.y / dist);
    atomicAdd(&d_dX[a].z, -strength * r.z / dist);
    atomicAdd(&d_dX[b].x, strength * r.x / dist);
    atomicAdd(&d_dX[b].y, strength * r.y / dist);
    atomicAdd(&d_dX[b].z, strength * r.z / dist);
}

__global__ void update_protrusions(const int n_cells,
    const Grid<n_max>* __restrict__ d_grid, const Cell* __restrict d_X,
    curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rand_nb_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
    auto b =
        d_grid->d_point_id[d_grid->d_cube_start[rand_nb_cube] +
                           min(static_cast<int>(
                                   curand_uniform(&d_state[i]) * cells_in_cube),
                               cells_in_cube - 1)];
    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto noise = curand_uniform(&d_state[i]);

    auto high_f = false;
    // auto high_f = (d_X[a].f + d_X[b].f) > 0.2f;
    auto distal = (d_X[a].f + d_X[b].f) > 0.025f;//0.025f;//0.20f; //0.025
    bool more_along_w = false;
    bool normal_to_f_gradient = false;
    bool normal_to_w = false;
    if(distal) {
        more_along_w =
            fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) * (1.f - noise);
        // normal_to_f_gradient =
        //     fabs(new_r.f / new_dist) < fabs(old_r.f / old_dist) * (1.f - noise);
        // normal_to_w =
        //     fabs(new_r.w / new_dist) < fabs(old_r.w / old_dist) * (1.f - noise);
        // high_f = true;
    } else {
        more_along_w =
            fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) * (1.f - noise);
        // normal_to_f_gradient =
        //     fabs(new_r.f / new_dist) < fabs(old_r.f / old_dist) * (1.f - noise);
        // high_f = true;
    }
    // high_f = false;
    // high_f = true;
    if (not_initialized or more_along_w or high_f or normal_to_f_gradient) {
        link->a = a;
        link->b = b;
    }
}

__global__ void proliferate(float max_rate, float mean_distance, Cell* d_X,
    int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * max_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - max_rate))
        return;  // Dividing new cells is problematic!

    // float rate = d_prolif_rate[i] * d_X[i].f;
    // float rate = d_prolif_rate[i] - d_prolif_rate[i]*(1.f - 0.25f)*(1.f-d_X[i].f);
    float rate = d_prolif_rate[i];


    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
            break;
        }
        // case epithelium: {
        //     // if (d_epi_nbs[i] > d_mes_nbs[i]) return;
        //     if (d_epi_nbs[i] > 7) return;
        //     if (d_mes_nbs[i] <= 0) return;
        //     auto r = curand_uniform(&d_state[i]);
        //     if (r > 2.5f * rate) return;  // 2.5
        // }
        default: {
            // if (d_epi_nbs[i] > d_mes_nbs[i]) return;
            if (d_epi_nbs[i] > 7) return;
            if (d_mes_nbs[i] <= 0) return;
            auto r = curand_uniform(&d_state[i]);
            if (r > 2.5f * rate) return;  // 2.5
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w / 2;
        d_X[i].w = d_X[i].w / 2;
        d_X[n].f = d_X[i].f / 2;
        d_X[i].f = d_X[i].f / 2;
    } else {
        d_X[n].w = d_X[i].w;
        d_X[n].f = d_X[i].f;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_prolif_rate[n] = d_prolif_rate[i];
    // d_mes_nbs[n] = 0;
    // d_epi_nbs[n] = 0;
}

// Double step solver means we have to initialise n_neibhbours before every
// step.
// This function is called before each step.
void neighbour_init(const Cell* __restrict__ d_X, Cell* d_dX)
{
    thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_max, 0);
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_max, 0);
}

template<int n_links, typename Pt = float3,
    Link_force<Pt> force = linear_force<Pt>>
void link_forces_w_n_init(
    Links<n_links>& links, const Pt* __restrict__ d_X, Pt* d_dX)
{
    thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_max, 0);
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_max, 0);
    link<Pt, force><<<(links.get_d_n() + 32 - 1) / 32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);
}

//*****************************************************************************

int main(int argc, char const* argv[])
{
    std::string file_name = argv[1];
    std::string output_tag = argv[2];
    float max_proliferation_rate = std::stof(argv[3]);
    int n_time_steps = std::stoi(argv[4]);
    int prolif_dist = std::stoi(argv[5]);

    // Load the initial conditions
    Vtk_input input(file_name);
    int n0 = input.n_bolls;
    Solution<Cell, n_max, Grid_solver> limb(n0);

    input.read_positions(limb);
    input.read_polarity(limb);

    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    Property<n_max, int> intype;

    input.read_property(intype, "cell_type");  // we read it as an int, then we translate to
                                               // enum "Cell_types"
    for (int i = 0; i < n0; i++) {
        limb.h_X[i].w = 0.0f;
        limb.h_X[i].f = 0.0f;
        if (intype.h_prop[i] == 0) {
            type.h_prop[i] = mesenchyme;
        } else if (intype.h_prop[i] == 1) {
            type.h_prop[i] = epithelium;
            limb.h_X[i].w = 1.0f;
        } else {
            type.h_prop[i] = aer;
            limb.h_X[i].w = 1.0f;
            limb.h_X[i].f = 1.0f;
        }
    }

    limb.copy_to_device();
    type.copy_to_device();

    std::cout << "initial nbolls " << n0 << " nmax " << n_max << std::endl;

    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    Links<static_cast<int>(n_max * prots_per_cell)> protrusions(
        protrusion_strength, n0 * prots_per_cell);
    auto intercalation = std::bind(
        link_forces_w_n_init<static_cast<int>(n_max * prots_per_cell), Cell, link_force>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    Grid<n_max> grid;

    // determine cell-specific proliferation rates
    Property<n_max, float> prolif_rate("prolif_rate");
    cudaMemcpyToSymbol(
        d_prolif_rate, &prolif_rate.d_prop, sizeof(d_prolif_rate));

    float min_proliferation_rate = 0.5f * max_proliferation_rate;
    // if (prolif_dist == 0) {
        for (int i = 0; i < n0; i++) {
            prolif_rate.h_prop[i] = max_proliferation_rate;
        }
    // } else {
    //     float xmax = -10000.0f;
    //     for (int i = 0; i < n0; i++) {
    //         if (limb.h_X[i].x > xmax) xmax = limb.h_X[i].x;
    //     }
    //     for (int i = 0; i < n0; i++) {
    //         if (limb.h_X[i].x < 0)
    //             prolif_rate.h_prop[i] = 0;
    //         else
    //             prolif_rate.h_prop[i] = min_proliferation_rate +
    //                                     pow((limb.h_X[i].x / xmax), 1) *
    //                                         max_proliferation_rate * 0.5f;
    //     }
    // }
    prolif_rate.copy_to_device();

    // State for proliferations
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(
        n_max, seed, d_state);

    int skip_step = 1;  // n_time_steps/10;
    std::cout << "n_time_steps " << n_time_steps << " write interval "
              << skip_step << std::endl;

    Vtk_output limb_output(output_tag);

    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if (time_step % skip_step == 0 || time_step == n_time_steps) {
            limb.copy_to_host();
            protrusions.copy_to_host();
            type.copy_to_host();
            n_epi_nbs.copy_to_host();
            n_mes_nbs.copy_to_host();
            prolif_rate.copy_to_host();
        }

        proliferate<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(
            max_proliferation_rate, r_min, limb.d_X, limb.d_n, d_state);
        protrusions.set_d_n(limb.get_d_n() * prots_per_cell);
        grid.build(limb, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            limb.get_d_n(), grid.d_grid, limb.d_X, protrusions.d_state,
            protrusions.d_link);

        limb.take_step<wall_force, wall_friction>(dt, intercalation);

        // write the output
        if (time_step % skip_step == 0 || time_step == n_time_steps) {
            limb_output.write_positions(limb);
            limb_output.write_links(protrusions);
            limb_output.write_polarity(limb);
            limb_output.write_field(limb, "Wint");
            limb_output.write_field(limb, "FGF", &Cell::f);
            limb_output.write_property(type);
            limb_output.write_property(n_epi_nbs);
            limb_output.write_property(n_mes_nbs);
            limb_output.write_property(prolif_rate);
        }
    }

    return 0;
}
