// Simulates branching on a spheroid induced by Turing mechanism on surface
#include <curand_kernel.h>
#include <time.h>
#include <thread>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"

const auto n_0 = 500;
const auto n_max = 500000;
const auto r_max = 1.0f;
const auto n_time_steps = 500;
const auto skip_steps = 10;
const auto dt = 0.2f;

// Turing parameters
const auto lambda = 0.0075;

const auto D_u = 0.001;
const auto D_v = 0.2;
const auto f_v = 1.0;
const auto f_u = 80.0;
const auto g_u = 80.0;
const auto m_u = 0.25;  // degradation rates
const auto m_v = 0.75;
const auto s_u = 0.05;

const auto epi_proliferation_rate = 0.2;
const auto mes_proliferation_rate = 0.1;
// Threshold conc. of v that allows mesench. cells to divide
const auto prolif_threshold = 1150.0f;

std::string output_tag = "branching_lineage_tracing_longeer";

enum Cell_types { mesenchyme, epithelium };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

// Cell lineage tracking
__device__ int* d_n_index;
__device__ int* d_cell_index;
__device__ int* d_node_parent;
__device__ int* d_cell_parent;
__device__ float3* d_node_coordinates;
__device__ float* d_node_time;
__device__ int* d_node_clone;
__device__ int* d_cell_clone;
__device__ Cell_types* d_node_type;

MAKE_PT(Cell, theta, phi, u, v);


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

        dF += bending_force(Xi, r, dist) * 0.2;
    } else {
        dF.v = -D_v * r.v;  // Diffuses into mesenchyme to induce proliferation
    }

    if (d_type[j] == epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    return dF;
}


__global__ void proliferate(float mean_distance, float time_progression, Cell* d_X, float3* d_old_v,
    int* d_n_cells, curandState* d_state)
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
            if (d_epi_nbs[i] > 5) return;

            if (d_mes_nbs[i] <= 0) return;

            if (rnd > epi_proliferation_rate) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
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
    d_old_v[n] = d_old_v[i];

    // // Cell lineage tracing
    auto tree_n = atomicAdd(d_n_index, 1);
    // printf("time prog %f i %i,tree_n %i",time_progression, i, tree_n);
    // save the coordinates of the cell that is going to become an internal node
    d_node_coordinates[tree_n].x = d_X[i].x;
    d_node_coordinates[tree_n].y = d_X[i].y;
    d_node_coordinates[tree_n].z = d_X[i].z;
    d_node_time[tree_n] = time_progression;
    // update parent cell
    d_node_parent[tree_n] = d_cell_parent[i];
    d_node_clone[tree_n] = d_cell_clone[i];
    d_node_type[tree_n] = d_type[i];
    // update cell indices for daughter cells
    d_cell_clone[n] = d_cell_clone[i];
    d_cell_parent[i] = tree_n;
    d_cell_parent[n] = tree_n;
}


int main(int argc, const char* argv[])
{
    // Initial state
    Solution<Cell, Grid_solver> cells{n_max, 100, r_max};
    *cells.h_n = n_0;
    relaxed_sphere(0.75, cells);
    Property<Cell_types> type{n_max, "type"};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) { type.h_prop[i] = mesenchyme; }
    cells.copy_to_device();
    type.copy_to_device();
    Property<int> n_mes_nbs{n_max, "n_mes_nbs"};
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<int> n_epi_nbs{n_max, "n_epi_nbs"};
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
    auto reset_nbs = [&](const int n, const Cell* __restrict__ d_X, Cell* d_dX) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);
    };
    curandState* d_state;  // For proliferations
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    // Cell lineage tracing
    Property<int> n_index{1, "n_index"};
    cudaMemcpyToSymbol(d_n_index, &n_index.d_prop, sizeof(d_n_index));

    Property<int> cell_index{n_max, "cell_index"};
    cudaMemcpyToSymbol(d_cell_index, &cell_index.d_prop, sizeof(d_cell_index));

    Property<int> cell_parent{n_max, "cell_parent"};
    cudaMemcpyToSymbol(d_cell_parent, &cell_parent.d_prop, sizeof(d_cell_parent));

    Property<int> node_parent{n_max, "node_parent"};
    cudaMemcpyToSymbol(d_node_parent, &node_parent.d_prop, sizeof(d_node_parent));

    Property<float3> node_coordinates{n_max, "node_coordinates"};
    cudaMemcpyToSymbol(d_node_coordinates, &node_coordinates.d_prop, sizeof(d_node_coordinates));

    Property<float> node_time{n_max, "node_time"};
    cudaMemcpyToSymbol(d_node_time, &node_time.d_prop, sizeof(d_node_time));

    Property<int> node_clone{2*n_max-1, "node_clone"};
    cudaMemcpyToSymbol(d_node_clone, &node_clone.d_prop, sizeof(d_node_clone));

    Property<Cell_types> node_type{2*n_max-1, "node_type"};
    cudaMemcpyToSymbol(d_node_type, &node_type.d_prop, sizeof(d_node_type));

    Property<int> cell_clone{n_max, "cell_clone"};
    cudaMemcpyToSymbol(d_cell_clone, &cell_clone.d_prop, sizeof(d_cell_clone));

    n_index.h_prop[0]=0;
    n_index.copy_to_device();


    // Find epithelium
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
    cells.take_step<epi_turing_mes_noturing>(0);
    cells.copy_to_host();
    n_mes_nbs.copy_to_host();
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 20) {
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                              cells.h_X[i].y * cells.h_X[i].y +
                              cells.h_X[i].z * cells.h_X[i].z);
            cells.h_X[i].theta = acosf(cells.h_X[i].z / dist);
            cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x);

            cells.h_X[i].u = rand() / (RAND_MAX + 1.) / 5 - 0.1;
            cells.h_X[i].v = rand() / (RAND_MAX + 1.) / 5 - 0.1;
        }
        cell_clone.h_prop[i] = i + 1;
        cell_parent.h_prop[i] = -1;
    }
    cells.copy_to_device();
    type.copy_to_device();
    cell_clone.copy_to_device();
    cell_parent.copy_to_device();

    // Integrate positions
    Vtk_output output{output_tag};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        type.copy_to_host();
        cell_clone.copy_to_host();

        std::thread calculation([&] {
            for (auto i = 0; i <= skip_steps; i++) {
                // printf("%i cells dn %i\n",time_step, cells.get_d_n());
                proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    0.75, float(time_step)/float(n_time_steps), cells.d_X, cells.d_old_v, cells.d_n, d_state);
                cells.take_step<epi_turing_mes_noturing>(dt, reset_nbs);
                // printf("%i after cells dn %i\n",time_step, cells.get_d_n());
            }
        });

        output.write_positions(cells);
        output.write_polarity(cells);
        output.write_field(cells, "u", &Cell::u);
        output.write_field(cells, "v", &Cell::v);
        output.write_property(type);
        output.write_property(cell_clone);

        calculation.join();
    }

    // Build the tree
    cells.copy_to_host();
    cell_index.copy_to_host();
    cell_parent.copy_to_host();
    cell_clone.copy_to_host();

    n_index.copy_to_host();
    node_coordinates.copy_to_host();
    node_time.copy_to_host();
    node_parent.copy_to_host();
    node_clone.copy_to_host();
    node_type.copy_to_host();

    auto n_tree = n_index.h_prop[0];
    Solution<Cell, Grid_solver> tree{n_tree + *cells.h_n};

    Links branches{n_tree + *cells.h_n, 0.0};

    // Internal nodes of the tree
    for(auto i=0 ; i<n_tree ; i++){
        tree.h_X[i].x = node_coordinates.h_prop[i].x;
        tree.h_X[i].y = node_coordinates.h_prop[i].y;
        tree.h_X[i].z = node_coordinates.h_prop[i].z;
        auto parent = node_parent.h_prop[i];
        // std::cout<<i<<" node parent "<<parent<<std::endl;
        if(parent >= 0){
            branches.h_link[i].a = i;
            branches.h_link[i].b = parent;
        }
    }

    // Terminal nodes of the tree (the current cells)
    for(auto i=0 ; i<*cells.h_n ; i++){
        tree.h_X[n_tree + i].x = cells.h_X[i].x;
        tree.h_X[n_tree + i].y = cells.h_X[i].y;
        tree.h_X[n_tree + i].z = cells.h_X[i].z;
        node_clone.h_prop[n_tree + i] = cell_clone.h_prop[i];
        node_type.h_prop[n_tree + i] = type.h_prop[i];

        node_time.h_prop[n_tree + i] = 1.0f;

        auto parent = cell_parent.h_prop[i];
        node_parent.h_prop[n_tree + i] = parent;
        // std::cout<<i<<" cell parent "<<parent<<std::endl;
        if(parent >= 0){
            branches.h_link[n_tree + i].a = n_tree + i;
            branches.h_link[n_tree + i].b = parent;
        }
    }

    Vtk_output tree_output{output_tag + ".tree", "output/", true};
    tree_output.write_positions(tree);
    tree_output.write_links(branches);
    tree_output.write_property(node_clone);
    tree_output.write_property(node_time);
    tree_output.write_property(node_parent);
    tree_output.write_property(node_type);

    return 0;
}
