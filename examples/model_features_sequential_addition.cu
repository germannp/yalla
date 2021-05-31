// Simulate mesenchymal intercalation orchestrated by epithelial signals
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/mesh.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1f;
const auto n_0 = 200;
const auto n_max = 4000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.25f;
const auto r_protrusion = 2.0f;
const auto proliferation_rate = 0.040f;
enum Cell_types { mesenchyme, epithelium };

MAKE_PT(Cell, w, theta, phi);


__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;

__device__ Cell force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        dF.w = -0.01f * (d_type[i] == mesenchyme) * Xi.w;
        if (Xi.w < 0.0f) dF.w = 0.0f;

        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.7 - dist, 0) * 3.0f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.7 - dist, 0) * 2.0f - fmaxf(dist - 0.8, 0);
    } else {
        F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 1.5f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.4f;
    if (Xi.w < 0.f) dF.w = 0.f;

    if (d_type[j] == mesenchyme)
        d_mes_nbs[i] += 1;
    else
        d_epi_nbs[i] += 1;

    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += bending_force(Xi, r, dist) * 0.10;
    return dF;
}


__global__ void proliferate(float rate, float mean_distance, Cell* d_X,
    float3* d_old_v, int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - rate))
        return;  // Dividing new cells is problematic!

    if(d_type[i] == mesenchyme) {
        auto rnd = curand_uniform(&d_state[i]);
        if (rnd > rate) return;
    }else{
        if (d_epi_nbs[i] > 14) return;
        if (d_mes_nbs[i] < 1) return;

        auto rnd = curand_uniform(&d_state[i]);
        if (rnd > 2.0f * rate) return;
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w / 2;
        d_X[i].w = d_X[i].w / 2;
    } else {
        d_X[n].w = d_X[i].w;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_old_v[n] = d_old_v[i];
}


__global__ void update_protrusions(const int n_cells,
    const Grid* __restrict__ d_grid, const Cell* __restrict d_X,
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

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];

    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto not_initialized = link->a == link->b;
    auto noise = curand_uniform(&d_state[i]);
    bool normal_to_w =
        fabs(new_r.w / new_dist) < fabs(old_r.w / old_dist) * (1.f - noise);

    if (not_initialized or normal_to_w) {
        link->a = a;
        link->b = b;
    }
}


int main(int argc, char const* argv[])
{
    // Prepare initial state
    Solution<Cell, Grid_solver> cells{n_max};
    *cells.h_n = n_0;
    random_sphere(0.55f, cells);

    Property<Cell_types> type{n_max};          // translate to Cell_types
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (int i = 0; i < n_0; i++) {
        cells.h_X[i].w = 0.0f;
        type.h_prop[i] = mesenchyme;
    }
    cells.copy_to_device();
    type.copy_to_device();

    Property<int> n_mes_nbs{n_max, "n_mes_nbs"};
    Property<int> n_epi_nbs{n_max, "n_epi_nbs"};
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    Vtk_output output{"model_features_sequential_addition"};

    // Part 1: Relax a sphere of mesenchymal cells
    for (auto time_step = 0; time_step <= 100; time_step++) {
        cells.copy_to_host();
        n_mes_nbs.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);

        cells.take_step<force, friction_on_background>(dt);

        output.write_positions(cells);
        output.write_polarity(cells);

        output.write_field(cells);
        output.write_property(type);
        output.write_property(n_mes_nbs);
    }

    n_mes_nbs.copy_to_host();
    cells.copy_to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 40){
            auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                cells.h_X[i].y * cells.h_X[i].y +
                cells.h_X[i].z * cells.h_X[i].z);
            cells.h_X[i].theta = acosf(cells.h_X[i].z / dist);
            cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x);
            type.h_prop[i] = epithelium;
        }
    }
    cells.copy_to_device();
    type.copy_to_device();

    // Part 2: Include epithelium surrounding the mesenchyme
    for (auto time_step = 0; time_step <= 100; time_step++) {
        cells.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);

        cells.take_step<force>(dt);

        output.write_positions(cells);
        output.write_polarity(cells);
        output.write_property(type);
        output.write_field(cells);
    }

    cells.copy_to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < n_0; i++) {
        if (cells.h_X[i].x > 1.0f)
            cells.h_X[i].w = 1.0f;
    }
    cells.copy_to_device();

    // Part 3: Add a morphogen gradient.
    for (auto time_step = 0; time_step <= 100; time_step++) {
        cells.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);

        cells.take_step<force>(dt);

        output.write_positions(cells);
        output.write_property(type);
        output.write_field(cells);
    }


    curandState* d_state;  // For proliferations
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    // Part 4: Add cell proliferation
    for (auto time_step = 0; time_step <= 100; time_step++) {
        cells.copy_to_host();
        type.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);

        cells.take_step<force>(dt);

        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            proliferation_rate, r_min, cells.d_X, cells.d_old_v, cells.d_n,
            d_state);

        output.write_positions(cells);
        output.write_property(type);
        output.write_field(cells);
    }



    Links protrusions{n_max * prots_per_cell, protrusion_strength};
    protrusions.set_d_n(n_0 * prots_per_cell);
    auto intercalation = [&](const int n, const Cell* __restrict__ d_X, Cell* d_dX) {
        return link_forces(protrusions, d_X, d_dX);
    };
    Grid grid{n_max};

    // Part 5: Add cell intercalation. Cells orient their protrusions
    // within the plane perpendicular to the direction of the morphogen
    // gradient.
    for (auto time_step = 0; time_step <= 100; time_step++) {
        cells.copy_to_host();
        protrusions.copy_to_host();
        type.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);


        protrusions.set_d_n(cells.get_d_n() * prots_per_cell);
        grid.build(cells, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            cells.get_d_n(), grid.d_grid, cells.d_X, protrusions.d_state,
            protrusions.d_link);

        cells.take_step<force>(dt, intercalation);

        output.write_positions(cells);
        output.write_links(protrusions);
        output.write_property(type);
        output.write_field(cells);
    }

    return 0;
}
