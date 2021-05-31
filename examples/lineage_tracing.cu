// Simulate lineage tracing of a group of dividing cells
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
const auto prolif_rate = 0.005;
const auto n_0 = 5;
const auto n_max = 5000;
const auto n_time_steps = 1000;
const auto dt = 0.1;

// Cell lineage tracking
__device__ int* d_n_index;
__device__ int* d_cell_index;
__device__ int* d_node_parent;
__device__ int* d_cell_parent;
__device__ float3* d_node_coordinates;
__device__ int* d_node_clone;
__device__ int* d_cell_clone;


__device__ Po_cell relaxation_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    float F = fmaxf(0.8 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);

    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    return dF;
}


__global__ void proliferate(float rate, int n_cells, curandState* d_state,
    Po_cell* d_X, float3* d_old_v, int* d_n_cells)
{
    D_ASSERT(n_cells * rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;  // Dividing new cells is problematic!

    auto rnd = curand_uniform(&d_state[i]);
    if (rnd > rate) return;

    // Cell lineage tracking
    auto n = atomicAdd(d_n_cells, 1);
    auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
    d_X[n].x = d_X[i].x + mean_dist / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_dist / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_dist / 4 * cosf(theta);
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_old_v[n] = d_old_v[i];

    // Cell lineage tracing
    auto tree_n = atomicAdd(d_n_index, 1);
    // save the coordinates of the cell that is going to become an internal node
    d_node_coordinates[tree_n].x = d_X[i].x;
    d_node_coordinates[tree_n].y = d_X[i].y;
    d_node_coordinates[tree_n].z = d_X[i].z;
    // update parent cell
    d_node_parent[tree_n] = d_cell_parent[i];
    d_node_clone[tree_n] = d_cell_clone[i];
    // update cell indices for daughter cells
    d_cell_clone[n] = d_cell_clone[i];
    d_cell_parent[i] = tree_n;
    d_cell_parent[n] = tree_n;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Grid_solver> cells{n_max};
    *cells.h_n = n_0;
    // relaxed_sphere(mean_dist, cells);
    regular_rectangle(mean_dist, n_0, cells);

    curandState* d_state;
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

    Property<int> node_clone{n_max, "node_clone"};
    cudaMemcpyToSymbol(d_node_clone, &node_clone.d_prop, sizeof(d_node_clone));

    Property<int> cell_clone{n_max, "cell_clone"};
    cudaMemcpyToSymbol(d_cell_clone, &cell_clone.d_prop, sizeof(d_cell_clone));


    // Set up indices for cell lineage tracking
    for (auto i = 0; i < n_0; i++){
        cell_parent.h_prop[i] = -1;
        node_parent.h_prop[i] = -1;
        cell_clone.h_prop[i] = i;
        node_clone.h_prop[i] = i;
        node_coordinates.h_prop[i].x = cells.h_X[i].x;
        node_coordinates.h_prop[i].y = cells.h_X[i].y;
        node_coordinates.h_prop[i].z = cells.h_X[i].z;
    }

    n_index.h_prop[0] = n_0;
    n_index.copy_to_device();
    cell_index.copy_to_device();
    cell_parent.copy_to_device();
    node_coordinates.copy_to_device();
    cell_clone.copy_to_device();
    node_clone.copy_to_device();

    // Simulate growth
    Vtk_output output{"lineage_tracing", "output/", false};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {

        cells.take_step<relaxation_force>(dt);
        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            prolif_rate * (time_step > 100), cells.get_d_n(), d_state,
            cells.d_X, cells.d_old_v, cells.d_n);

        cudaDeviceSynchronize();

        cells.copy_to_host();
        cell_index.copy_to_host();
        cell_parent.copy_to_host();
        cell_clone.copy_to_host();

        output.write_positions(cells);
        output.write_polarity(cells);
        output.write_property(cell_index);
        output.write_property(cell_parent);
        output.write_property(cell_clone);
    }

    // Build the tree
    cells.copy_to_host();
    cell_index.copy_to_host();
    cell_parent.copy_to_host();
    cell_clone.copy_to_host();

    n_index.copy_to_host();
    node_coordinates.copy_to_host();
    node_parent.copy_to_host();
    node_clone.copy_to_host();

    auto n_tree = n_index.h_prop[0];
    Solution<Po_cell, Grid_solver> tree{n_tree + *cells.h_n};

    Links branches{n_tree + *cells.h_n, 0.0};

    // Internal nodes of the tree
    for(auto i=0 ; i<n_tree ; i++){
        tree.h_X[i].x = node_coordinates.h_prop[i].x;
        tree.h_X[i].y = node_coordinates.h_prop[i].y;
        tree.h_X[i].z = node_coordinates.h_prop[i].z;
        auto parent = node_parent.h_prop[i];
        std::cout<<i<<" node parent "<<parent<<std::endl;
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

        auto parent = cell_parent.h_prop[i];
        std::cout<<i<<" cell parent "<<parent<<std::endl;
        if(parent >= 0){
            branches.h_link[n_tree + i].a = n_tree + i;
            branches.h_link[n_tree + i].b = parent;
        }
    }

    Vtk_output tree_output{"lineage_tree", "output/", false};
    tree_output.write_positions(tree);
    tree_output.write_links(branches);
    tree_output.write_property(node_clone);

    return 0;
}
