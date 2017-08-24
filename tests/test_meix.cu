
#include <assert.h>
#include <curand_kernel.h>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

#include "../models/meix/meix.h"
#include "minunit.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1;
const auto n_max = 150000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;

enum Cell_types { mesenchyme, epithelium };

__device__ Cell_types* d_type;

MAKE_PT(Cell, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) return dF;

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 2.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    if (d_type[i] == epithelium && d_type[j] == epithelium)
        dF += rigidity_force(Xi, r, dist) * 0.10f;

    return dF;
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
    auto new_one = curand_uniform(&d_state[i]) < 0.05f;
    if (not_initialized || new_one) {
        link->a = a;
        link->b = b;
    }
}

__device__ float relaxation_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    return 0;
}

// Distribute bolls uniformly random in rectangular cube
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_cubic_rectangle(float xmin, float ymin, float zmin, float dx,
    float dy, float dz, Solution<Pt, n_max, Solver>& bolls,
    unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);

    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = xmin + dx * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].y = ymin + dy * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].z = zmin + dz * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].phi = 0.0f;
        bolls.h_X[i].theta = 0.0f;
    }

    bolls.copy_to_device();
}

int main(int argc, char const* argv[])
{

    // First, load the mesh file so we can get the maximum dimensions of the
    // system
    Meix meix("/home/mmarin/Desktop/Limb_project_data/Donut/DonutR1r05.vtk");

    // meix defines the overall shape of the limb bud (mesench. + ectoderm)
    float resc = 5.f;
    meix.Rescale_relative(resc);

    // meix_mesench defines the volume occupied by the mesenchyme (smaller than
    // meix)
    Meix meix_mesench = meix;
    meix_mesench.Rescale_absolute(-r_min);  //*1.3//*1.2

    // Compute min. and max, positions in x,y,z from rescaled mesh
    float xmin = 10000.0f;
    float xmax = -10000.0f;
    float ymin = 10000.0f;
    float ymax = -10000.0f;
    float zmin = 10000.0f;
    float zmax = -10000.0f;
    for (int i = 0; i < meix.n_vertices; i++) {
        if (meix.Vertices[i].x < xmin)
            xmin = meix.Vertices[i].x;
        if (meix.Vertices[i].x > xmax)
            xmax = meix.Vertices[i].x;
        if (meix.Vertices[i].y < ymin)
            ymin = meix.Vertices[i].y;
        if (meix.Vertices[i].y > ymax)
            ymax = meix.Vertices[i].y;
        if (meix.Vertices[i].z < zmin)
            zmin = meix.Vertices[i].z;
        if (meix.Vertices[i].z > zmax)
            zmax = meix.Vertices[i].z;
    }
    float dx = xmax - xmin;
    float dy = ymax - ymin;
    float dz = zmax - zmin;

    // we use the maximum lengths of the mesh to draw a cube that includes the
    // mesh
    // Let's fill the cube with bolls

    // Now we include intercalation in the cubic relaxation, so we must assume a
    // larger cube, since the end result will be compressed to some extent
    float factor = 0.1f;
    float r = dx * factor / 2;
    float new_xmin = xmin - r;
    r = dy * factor / 2;
    float new_ymin = ymin - r;
    r = dz * factor / 2;
    float new_zmin = zmin - r;
    float new_dx = dx + dx * factor, new_dy = dy + dy * factor,
          new_dz = dz + dz * factor;

    float cube_vol = new_dx * new_dy * new_dz;
    float r_boll = 0.5f * r_min;
    float boll_vol = 4.f / 3.f * M_PI * pow(r_boll, 3);
    int n_bolls_cube = cube_vol / boll_vol;

    Solution<Cell, n_max, Grid_solver> cube(n_bolls_cube);

    // Fill the cube with bolls
    uniform_cubic_rectangle(
        new_xmin, new_ymin, new_zmin, new_dx, new_dy, new_dz, cube);

    // Variable indicating cell type
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));

    cube.copy_to_device();
    type.copy_to_device();

    // Declaration of links
    Links<static_cast<int>(n_max * prots_per_cell)> protrusions(
        protrusion_strength, n_bolls_cube * prots_per_cell);
    auto intercalation =
    std::bind(link_forces<static_cast<int>(n_max * prots_per_cell), Cell>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    Grid<n_max> grid;

    // State for links
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(d_state, n_max);

    for (auto time_step = 0; time_step <= 1000; time_step++)
        cube.take_step<relaxation_force, relaxation_friction>(dt);

    std::cout<<"Cube 1 integrated"<<std::endl;

    // The relaxed cube positions will be used to imprint epithelial cells
    cube.copy_to_host();
    std::vector<Point> cube_relax_points;
    for (auto i = 0; i < n_bolls_cube; i++) {
        Point p = Point(cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z);
        cube_relax_points.push_back(p);
    }

    // We apply the links to the relaxed cube to compress it (as will be the
    // mesench in the limb bud)
    for (auto time_step = 0; time_step <= 1000; time_step++) {
        protrusions.set_d_n(cube.get_d_n() * prots_per_cell);
        grid.build(cube, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            cube.get_d_n(), grid.d_grid, cube.d_X, protrusions.d_state,
            protrusions.d_link);

        cube.take_step<relaxation_force, relaxation_friction>(
            dt, intercalation);

    }
    std::cout
        <<"Cube 2 integrated with links (only when links flag is active)"
        <<std::endl;

    // Fit the cube into a mesh and sort which cells are inside the mesh
    // For the mesenchyme we use the smaller mesh and the compressed cube
    // For the epithelium we use the larger meix and the relaxed cube

    // Mesenchyme
    // Setup the list of points
    std::vector<Point> cube_points;
    for (auto i = 0; i < n_bolls_cube; i++) {
        Point p = Point(cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z);
        cube_points.push_back(p);
    }

    // Setup the list of inclusion test results
    int* mesench_result = new int[n_bolls_cube];
    // Set direction of ray
    Point dir = Point(0.0f, 1.0f, 0.0f);

    meix_mesench.InclusionTest(cube_points, mesench_result, dir);

    // Make a new list with the ones that are inside
    std::vector<Point> mes_cells;
    int n_bolls_mes = 0;
    for (int i = 0; i < n_bolls_cube; i++) {
        if (mesench_result[i] == 1) {
            mes_cells.push_back(cube_points[i]);
            n_bolls_mes++;
        }
    }

    std::cout << "bolls_in_cube " << n_bolls_cube << " bolls after fill "
              << n_bolls_mes << std::endl;

    // Epithelium (we have to sort out which ones are inside the big mesh and
    // out of the small one)
    // Setup the list of inclusion test results
    int* epi_result_big = new int[n_bolls_cube];
    int* epi_result_small = new int[n_bolls_cube];

    meix.InclusionTest(cube_relax_points, epi_result_big, dir);
    meix_mesench.InclusionTest(cube_relax_points, epi_result_small, dir);

    // Make a new list with the ones that are inside
    std::vector<Point> epi_cells;
    int n_bolls_epi = 0;
    for (int i = 0; i < n_bolls_cube; i++) {
        if (epi_result_big[i] == 1 and epi_result_small[i] == 0) {
            epi_cells.push_back(cube_relax_points[i]);
            n_bolls_epi++;
        }
    }

    int n_bolls_total = n_bolls_mes + n_bolls_epi;

    std::cout << "bolls_in_mes " << n_bolls_mes << " bolls_in_epi "
              << n_bolls_epi << " bolls_in_total " << n_bolls_total
              << std::endl;

    Solution<Cell, n_max, Grid_solver> bolls(n_bolls_total);

    for (int i = 0; i < n_bolls_mes; i++) {
        bolls.h_X[i].x = mes_cells[i].x;
        bolls.h_X[i].y = mes_cells[i].y;
        bolls.h_X[i].z = mes_cells[i].z;
        type.h_prop[i] = mesenchyme;
    }
    int count = 0;
    for (int i = n_bolls_mes; i < n_bolls_total; i++) {
        bolls.h_X[i].x = epi_cells[count].x;
        bolls.h_X[i].y = epi_cells[count].y;
        bolls.h_X[i].z = epi_cells[count].z;
        type.h_prop[i] = epithelium;
        // polarity
        Point p = epi_cells[count];
        int f = -1;
        float dmin = 1000000.f;
        // we use the closest facet on meix to determine the polarity of the
        // epithelial cell
        for (int j = 0; j < meix.n_facets; j++) {
            Point r = p - meix.Facets[j].C;
            float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
            if (d < dmin) {
                dmin = d;
                f = j;
            }
        }
        count++;
        bolls.h_X[i].phi = atan2(meix.Facets[f].N.y, meix.Facets[f].N.x);
        bolls.h_X[i].theta = acos(meix.Facets[f].N.z);
    }

    bolls.copy_to_device();
    type.copy_to_device();

    Vtk_output output("donut_test");

    bolls.copy_to_host();
    output.write_positions(bolls);
    output.write_polarity(bolls);
    output.write_property(type);

    meix.WriteVtk("trololol");
    meix_mesench.WriteVtk("trololol_mesench");

    //test with the torus equation (R-sqrt(x^2+y^2)+z^2 <= r^2)
    bolls.copy_to_host();
    float big_R = resc;
    float small_r = 0.5f*resc;
    for (int i = 0 ; i < n_bolls_total ; i++) {
        float test = pow(big_R - sqrt(pow(bolls.h_X[i].x, 2) +
            pow(bolls.h_X[i].y, 2)), 2) + pow(bolls.h_X[i].z, 2) -
            pow(small_r, 2);
            if(test>0)
                std::cout<<"n_bolls "<<n_bolls_total<<" i "<<i<<" test "<<test<<std::endl;
            assert (test < 0.f or fabs(test < 0.1f));
        // MU_ASSERT("lol.to_cstring()", (test < 0.1f));
    }

    std::cout << "DOOOOOOOOOOOOOOONE***************" << std::endl;

    return 0;
}
