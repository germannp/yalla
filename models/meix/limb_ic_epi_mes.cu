// Makes initial conditions for a limb bud taking the morphology from a 3D model
//(3D mesh), then fills the volume with mesenchymal cells and the surface with
// epithelial cells, then lets teh system relax.

// Command line arguments
// argv[1]=input mesh file name
// argv[2]=output file tag
// argv[3]=target limb bud size (dx)
// argv[4]=cube relax_time
// argv[5]=limb bud relax_time
// argv[6]=links flag (activate if you want to use links in later simulations)
// argv[7]=wall flag (activate in limb buds, when you want a wall boundary cond.).
// argv[8]=AER flag (activate in limb buds)

#include <curand_kernel.h>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/links.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"

#include "meix.h"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1;
const auto n_max = 150000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;

enum Cell_types { mesenchyme, epithelium, aer };

__device__ Cell_types* d_type;
__device__ int* d_freeze;

MAKE_PT(Cell, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) return dF;

    if (d_freeze[i] == 1)
        return dF;  // frozen cells don't experience force so don't move

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

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += rigidity_force(Xi, r, dist) * 0.10f;

    return dF;
}

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j)
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

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += rigidity_force(Xi, r, dist) * 0.5f;

    if (Xi.x < 0) dF.x = 0.f;

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

__device__ float freeze_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (d_freeze[i] == 1) return 0;
    return 1;
}

__device__ float wall_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (Xi.x < 0) return 0;
    return 1;
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

template<typename Pt, int n_max, template<typename, int> class Solver>
void fill_solver_w_meix_no_flank(
    Meix meix, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    // eliminate the flank boundary
    int i = 0;
    while (i < meix.Facets.size()) {
        if (meix.Facets[i].N.x > -1.01f && meix.Facets[i].N.x < -0.99f)
            meix.Facets.erase(meix.Facets.begin() + i);
        else
            i++;
    }
    meix.n_facets = meix.Facets.size();

    *bolls.h_n = meix.n_facets;
    assert(n_0 < *bolls.h_n);

    for (int j = 0; j < meix.n_facets; j++) {
        Triangle T = meix.Facets[j];
        float r = sqrt(pow(T.N.x, 2) + pow(T.N.y, 2) + pow(T.N.z, 2));
        bolls.h_X[j].x = T.C.x;
        bolls.h_X[j].y = T.C.y;
        bolls.h_X[j].z = T.C.z;
        bolls.h_X[j].phi = atan2(T.N.y, T.N.x);
        bolls.h_X[j].theta = acos(T.N.z / r);
    }
}

template<typename Pt, int n_max, template<typename, int> class Solver,
    typename Prop>
void fill_solver_w_epithelium(Solution<Pt, n_max, Solver>& inbolls,
    Solution<Pt, n_max, Solver>& outbolls, Prop& type, unsigned int n_0 = 0)
{
    assert(n_0 < *inbolls.h_n);
    assert(n_0 < *outbolls.h_n);

    int j = 0;
    for (int i = 0; i < *inbolls.h_n; i++) {
        if (type.h_prop[i] == epithelium) {
            outbolls.h_X[j].x = inbolls.h_X[i].x;
            outbolls.h_X[j].y = inbolls.h_X[i].y;
            outbolls.h_X[j].z = inbolls.h_X[i].z;
            outbolls.h_X[j].phi = inbolls.h_X[i].phi;
            outbolls.h_X[j].theta = inbolls.h_X[i].theta;
            j++;
        }
    }
    *outbolls.h_n = j;
}


int main(int argc, char const* argv[])
{

    std::string file_name = argv[1];
    std::string output_tag = argv[2];
    float target_dx = std::stof(argv[3]);
    int cube_relax_time = std::stoi(argv[4]);
    int epi_relax_time = std::stoi(argv[5]);
    bool links_flag = false;
    if(stoi(argv[6]) == 1)
        links_flag = true;
    bool wall_flag = false;
    if(stoi(argv[7]) == 1)
        wall_flag = true;
    bool AER_flag = false;
    if(stoi(argv[8]) == 1)
        AER_flag = true;

    // First, load the mesh file so we can get the maximum dimensions of the
    // system
    Meix meix(file_name);

    // Compute max length in X axis to know how much we need to rescale
    //**********************************************************************
    // Attention! we are assuming the PD axis of the limb is aligned with X
    //**********************************************************************

    float xmin = 10000.0f, xmax = -10000.0f;
    float ymin, ymax, zmin, zmax;
    float dx, dy, dz;

    for (int i = 0; i < meix.n_vertices; i++) {
        if (meix.Vertices[i].x < xmin) xmin = meix.Vertices[i].x;
        if (meix.Vertices[i].x > xmax) xmax = meix.Vertices[i].x;
    }
    dx = xmax - xmin;

    float resc = target_dx / dx;
    std::cout << "xmax= " << xmax << " xmin= " << xmin << std::endl;
    std::cout << "dx= " << dx << " target_dx= " << target_dx
              << " rescaling factor resc= " << resc << std::endl;


    // meix defines the overall shape of the limb bud (mesench. + ectoderm)
    meix.Rescale_relative(resc);
    // meix_mesench defines the volume occupied by the mesenchyme (smaller than
    // meix)
    Meix meix_mesench = meix;
    meix_mesench.Rescale_absolute(-r_min, wall_flag);  //*1.3//*1.2
    // Compute min. and max, positions in x,y,z from rescaled mesh
    xmin = 10000.0f;
    xmax = -10000.0f;
    ymin = 10000.0f;
    ymax = -10000.0f;
    zmin = 10000.0f;
    zmax = -10000.0f;
    for (int i = 0; i < meix_mesench.n_vertices; i++) {
        if (meix_mesench.Vertices[i].x < xmin)
            xmin = meix_mesench.Vertices[i].x;
        if (meix_mesench.Vertices[i].x > xmax)
            xmax = meix_mesench.Vertices[i].x;
        if (meix_mesench.Vertices[i].y < ymin)
            ymin = meix_mesench.Vertices[i].y;
        if (meix_mesench.Vertices[i].y > ymax)
            ymax = meix_mesench.Vertices[i].y;
        if (meix_mesench.Vertices[i].z < zmin)
            zmin = meix_mesench.Vertices[i].z;
        if (meix_mesench.Vertices[i].z > zmax)
            zmax = meix_mesench.Vertices[i].z;
    }
    dx = xmax - xmin;
    dy = ymax - ymin;
    dz = zmax - zmin;

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

    std::cout << "cube dims " << dx << " " << dy << " " << dz << std::endl;
    std::cout << "cube_vol " << cube_vol << std::endl;
    std::cout << "r_boll " << r_boll << std::endl;
    std::cout << "boll_vol " << boll_vol << std::endl;
    std::cout << "nbolls in cube " << n_bolls_cube << std::endl;

    Solution<Cell, n_max, Grid_solver> cube(n_bolls_cube);
    // Fill the cube with bolls
    uniform_cubic_rectangle(
        new_xmin, new_ymin, new_zmin, new_dx, new_dy, new_dz, cube);

    // Variable indicating cell type
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    // Variable that indicates which cells are 'frozen', so don't move
    Property<n_max, int> freeze("freeze");
    cudaMemcpyToSymbol(d_freeze, &freeze.d_prop, sizeof(d_freeze));

    for (auto i = 0; i < n_bolls_cube; i++) {
        type.h_prop[i] = mesenchyme;
        freeze.h_prop[i] = 0;
    }

    cube.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

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

    // Relaxation of the cube
    int skip_step = 1;  // relax_time/10;
    // std::cout<<"relax_time "<<relax_time<<" write interval "<<
    // skip_step<<std::endl;

    // Vtk_output cubic_output1(output_tag+".cubic_relaxation1");

    for (auto time_step = 0; time_step <= cube_relax_time; time_step++) {
        // if(time_step%skip_step==0 || time_step==cube_relax_time){
        //     cube.copy_to_host();
        // }

        cube.take_step<relaxation_force, relaxation_friction>(dt);

        // write the output
        // if(time_step%skip_step==0 || time_step==cube_relax_time) {
        //     cubic_output1.write_positions(cube);
        // }
    }

    std::cout<<"Cube 1 integrated"<<std::endl;

    // The relaxed cube positions will be used to imprint epithelial cells
    cube.copy_to_host();
    std::vector<Point> cube_relax_points;
    for (auto i = 0; i < n_bolls_cube; i++) {
        Point p = Point(cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z);
        cube_relax_points.push_back(p);
    }

    if(links_flag) {

        // Vtk_output cubic_output(output_tag+".cubic_relaxation");

        // We apply the links to the relaxed cube to compress it (as will be the
        // mesench in the limb bud)
        for (auto time_step = 0; time_step <= cube_relax_time; time_step++) {
            // if(time_step%skip_step==0 || time_step==cube_relax_time){
            //     cube.copy_to_host();
            //     protrusions.copy_to_host();
            // }

            protrusions.set_d_n(cube.get_d_n() * prots_per_cell);
            grid.build(cube, r_protrusion);
            update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
                cube.get_d_n(), grid.d_grid, cube.d_X, protrusions.d_state,
                protrusions.d_link);

            cube.take_step<relaxation_force, relaxation_friction>(
                dt, intercalation);

            // write the output
            // if(time_step%skip_step==0 || time_step==cube_relax_time) {
            //     cubic_output.write_positions(cube);
            //     cubic_output.write_links(protrusions);
            // }
        }
        std::cout
            <<"Cube 2 integrated with links (only when links flag is active)"
            <<std::endl;
    }

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
        freeze.h_prop[i] = 1;
    }
    int count = 0;
    for (int i = n_bolls_mes; i < n_bolls_total; i++) {
        bolls.h_X[i].x = epi_cells[count].x;
        bolls.h_X[i].y = epi_cells[count].y;
        bolls.h_X[i].z = epi_cells[count].z;
        type.h_prop[i] = epithelium;
        freeze.h_prop[i] = 0;
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
        if (meix.Facets[f].C.x < 0.1f && wall_flag) {  // the cells contacting the flank
                                                       // boundary can't be epithelial 0.001
            type.h_prop[i] = mesenchyme;
            freeze.h_prop[i] = 1;
            continue;
        }
        bolls.h_X[i].phi = atan2(meix.Facets[f].N.y, meix.Facets[f].N.x);
        bolls.h_X[i].theta = acos(meix.Facets[f].N.z);
    }
    std::cout << "count " << count << " epi_cells " << n_bolls_epi << std::endl;

    bolls.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

    std::cout << "n_bolls_total= " << n_bolls_total << std::endl;

    if(AER_flag) {
        // Imprint the AER on the epithelium (based on a mesh file too)
        std::string AER_file=file_name;
        AER_file.insert(AER_file.length() - 4, "_AER");
        std::cout<<"AER file "<<AER_file<<std::endl;
        Meix AER(AER_file);
        AER.Rescale_relative(resc);

        for (int i = n_bolls_mes; i < n_bolls_total; i++) {
            Point p(bolls.h_X[i].x, bolls.h_X[i].y, bolls.h_X[i].z);
            for (int j = 0; j < AER.n_facets; j++) {
                Point r = p - AER.Facets[j].C;
                float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
                if (d < r_min*1.5f) {
                    type.h_prop[i]=aer;
                    break;
                }
            }
        }

        AER.WriteVtk(output_tag + ".aer");
    }

    Vtk_output output(output_tag);

    skip_step = 1;  // relax_time/10;
    for (auto time_step = 0; time_step <= epi_relax_time; time_step++) {
        if (time_step % skip_step == 0 || time_step == epi_relax_time) {
            bolls.copy_to_host();
        }

        bolls.take_step<relaxation_force, freeze_friction>(dt);

        // write the output
        if (time_step % skip_step == 0 || time_step == epi_relax_time) {
            output.write_positions(bolls);
            output.write_polarity(bolls);
            output.write_property(type);
            output.write_property(freeze);
        }
    }

    // bolls.copy_to_device();
    // output.write_positions(bolls);
    // output.write_polarity(bolls);
    // output.write_property(type);

    // Unfreeze the mesenchyme
    for (int i = 0; i < n_bolls_total; i++) {
        if (type.h_prop[i] == mesenchyme) freeze.h_prop[i] = 0;
    }

    // write down the meix in the vtk file to compare it with the posterior
    // seeding
    meix.WriteVtk(output_tag);
    // write down the mesenchymal mesh in the vtk file to compare it with the
    // posterior filling
    meix_mesench.WriteVtk(output_tag + ".mesench");

    // Create a dummy meix that depicts the x=0 plane, depicting the flank
    // boundary
    Meix wall;
    Point A(0.f, 2 * ymin, 2 * zmin);
    Point B(0.f, 2 * ymin, 2 * zmax);
    Point C(0.f, 2 * ymax, 2 * zmin);
    Point D(0.f, 2 * ymax, 2 * zmax);
    Point N(1.f, 0.f, 0.f);
    Triangle ABC(A, B, C, N);
    Triangle BCD(B, C, D, N);
    wall.n_facets = 2;
    wall.Facets.push_back(ABC);
    wall.Facets.push_back(BCD);
    wall.WriteVtk(output_tag + ".wall");

    // for shape comparison purposes we write down the initial mesh as the
    // facets
    // centres and the bolls epithelium in separate vtk files.

    std::cout << "writing meix_T0" << std::endl;
    Solution<Cell, n_max, Grid_solver> meix_T0(meix.n_facets);
    fill_solver_w_meix_no_flank(meix, meix_T0);
    Vtk_output output_meix_T0(output_tag + ".meix_T0");
    output_meix_T0.write_positions(meix_T0);
    output_meix_T0.write_polarity(meix_T0);

    std::cout << "writing epi_T0" << std::endl;
    Solution<Cell, n_max, Grid_solver> epi_T0(n_bolls_total);
    fill_solver_w_epithelium(bolls, epi_T0, type);
    Vtk_output output_epi_T0(output_tag + ".epi_T0");
    output_epi_T0.write_positions(epi_T0);
    output_epi_T0.write_polarity(epi_T0);

    std::cout << "DOOOOOOOOOOOOOOONE***************" << std::endl;

    return 0;
}
