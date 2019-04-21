#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


__device__ float4 oscillator(float4 Xi, float4 r, float dist, int i, int j)
{
    float4 dF{0};
    if (i == j) return dF;

    if (i == 0) return Xi - r;

    return -(Xi - r);
}

const char* test_oscillation()
{
    Solution<float4, Tile_solver> oscillation{2};
    oscillation.h_X[0].w = 1;
    oscillation.h_X[1].w = 0;
    oscillation.copy_to_device();

    auto n_steps = 100;
    for (auto i = 0; i < n_steps; i++) {
        oscillation.take_step<oscillator>(2 * M_PI / n_steps);
        oscillation.copy_to_host();
        MU_ASSERT("Oscillator off circle",
            isclose(
                powf(oscillation.h_X[0].w, 2) + powf(oscillation.h_X[1].w, 2),
                1));
    }
    oscillation.copy_to_host();
    MU_ASSERT("Oscillator final cosine", isclose(oscillation.h_X[0].w, 1));
    // The sine is substantially less precise ;-)

    return NULL;
}


const auto L_0 = 0.5;

__device__ float3 clipped_spring(float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist >= 1) return dF;

    dF = r * (L_0 - dist) / dist;
    return dF;
}

const char* test_tile_tetrahedron()
{
    Solution<float3, Tile_solver> tile{4};
    random_sphere(L_0, tile);
    auto com_i = center_of_mass(tile);
    for (auto i = 0; i < 500; i++) { tile.take_step<clipped_spring>(0.1); }

    tile.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto r = tile.h_X[0] - tile.h_X[i];
        auto dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        MU_ASSERT("Spring not relaxed in tile tetrahedron", isclose(dist, L_0));
    }

    auto com_f = center_of_mass(tile);
    MU_ASSERT("Momentum in tile tetrahedron", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in tile tetrahedron", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in tile tetrahedron", isclose(com_i.z, com_f.z));

    return NULL;
}

const char* test_grid_tetrahedron()
{
    Solution<float3, Grid_solver> grid{4};
    random_sphere(L_0, grid);
    auto com_i = center_of_mass(grid);
    for (auto i = 0; i < 500; i++) { grid.take_step<clipped_spring>(0.1); }

    grid.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto r = float3{grid.h_X[0].x - grid.h_X[i].x,
            grid.h_X[0].y - grid.h_X[i].y, grid.h_X[0].z - grid.h_X[i].z};
        auto dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        MU_ASSERT("Spring not relaxed in grid tetrahedron", isclose(dist, L_0));
    }

    auto com_f = center_of_mass(grid);
    MU_ASSERT("Momentum in grid tetrahedron", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in grid tetrahedron", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in grid tetrahedron", isclose(com_i.z, com_f.z));

    return NULL;
}

const auto n_max = 50;

const char* test_compare_methods()
{
    Solution<float3, Tile_solver> tile{n_max};
    Solution<float3, Grid_solver> grid{n_max};
    random_sphere(0.733333, tile);
    for (auto i = 0; i < n_max; i++) {
        grid.h_X[i].x = tile.h_X[i].x;
        grid.h_X[i].y = tile.h_X[i].y;
        grid.h_X[i].z = tile.h_X[i].z;
    }
    grid.copy_to_device();
    for (auto i = 0; i < 2; i++) tile.take_step<clipped_spring>(0.1);
    for (auto i = 0; i < 2; i++) grid.take_step<clipped_spring>(0.1);

    tile.copy_to_host();
    grid.copy_to_host();
    for (auto i = 0; i < n_max; i++) {
        MU_ASSERT("Methods disagree", isclose(tile.h_X[i].x, grid.h_X[i].x));
        MU_ASSERT("Methods disagree", isclose(tile.h_X[i].y, grid.h_X[i].y));
        MU_ASSERT("Methods disagree", isclose(tile.h_X[i].z, grid.h_X[i].z));
    }

    return NULL;
}


__device__ float3 no_pw_int(float3 Xi, float3 r, float dist, int i, int j)
{
    return float3{0};
}

__global__ void push_cell(float3* d_dX)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) return;

    d_dX[1] = float3{1, 0, 0};
}

void push(const float3* __restrict__ d_X, float3* d_dX)
{
    push_cell<<<1, 1>>>(d_dX);
}

const char* test_generic_forces()
{
    Solution<float3, Tile_solver> tile{2};
    tile.h_X[0] = float3{0, 0, 10};
    tile.h_X[1] = float3{0, 0, 0};
    tile.copy_to_device();
    auto com_i = center_of_mass(tile);
    tile.take_step<no_pw_int>(1, push);

    tile.copy_to_host();
    auto com_f = center_of_mass(tile);
    MU_ASSERT("Momentum in tile generic force", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in tile generic force", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in tile generic force", isclose(com_i.z, com_f.z));

    MU_ASSERT("Tile generic force failed in x", isclose(tile.h_X[1].x, 0.5));
    MU_ASSERT("Tile generic force failed in y", isclose(tile.h_X[1].y, 0));
    MU_ASSERT("Tile generic force failed in z", isclose(tile.h_X[1].z, 0));

    Solution<float3, Grid_solver> grid{2};
    grid.h_X[0] = float3{0, 0, 10};
    grid.h_X[1] = float3{0, 0, 0};
    grid.copy_to_device();
    com_i = center_of_mass(grid);
    grid.take_step<clipped_spring>(1, push);

    grid.copy_to_host();
    com_f = center_of_mass(grid);
    MU_ASSERT("Momentum in grid generic force", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in grid generic force", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in grid generic force", isclose(com_i.z, com_f.z));

    MU_ASSERT("Grid generic force failed in x", isclose(grid.h_X[1].x, 0.5));
    MU_ASSERT("Grid generic force failed in y", isclose(grid.h_X[1].y, 0));
    MU_ASSERT("Grid generic force failed in z", isclose(grid.h_X[1].z, 0));

    return NULL;
}


const char* test_friction()
{
    Solution<float3, Tile_solver> tile{2};
    tile.h_X[0] = float3{0, 0, 0};
    tile.h_X[1] = float3{.5, 0, 0};
    tile.copy_to_device();
    for (auto i = 0; i < 10; i++)
        tile.take_step<no_pw_int, friction_on_background>(0.05, push);
    tile.copy_to_host();
    MU_ASSERT("Tile friction on background",
        isclose(tile.h_X[1].x - tile.h_X[0].x, 1));

    tile.h_X[0] = float3{0, 0, 0};
    tile.h_X[1] = float3{.5, 0, 0};
    tile.copy_to_device();
    for (auto i = 0; i < 10; i++) tile.take_step<no_pw_int>(0.05, push);
    tile.copy_to_host();
    MU_ASSERT("Tile friction w/ neighbour",
        isclose(tile.h_X[1].x - tile.h_X[0].x, 0.75));

    Solution<float3, Grid_solver> grid{2};
    grid.h_X[0] = float3{0, 0, 0};
    grid.h_X[1] = float3{.5, 0, 0};
    grid.copy_to_device();
    for (auto i = 0; i < 10; i++)
        grid.take_step<no_pw_int, friction_on_background>(0.05, push);
    grid.copy_to_host();
    MU_ASSERT("Grid friction on background",
        isclose(grid.h_X[1].x - grid.h_X[0].x, 1));

    grid.h_X[0] = float3{0, 0, 0};
    grid.h_X[1] = float3{.5, 0, 0};
    grid.copy_to_device();
    for (auto i = 0; i < 10; i++) grid.take_step<no_pw_int>(0.05, push);
    grid.copy_to_host();
    MU_ASSERT("Grid friction w/ neighbour",
        isclose(grid.h_X[1].x - grid.h_X[0].x, 0.75));

    return NULL;
}


const char* test_fix_point()
{
    Solution<float3, Tile_solver> tile{100};
    random_sphere(0.733333, tile);
    auto fix_point = 13;
    tile.h_X[fix_point] = float3{0};
    tile.copy_to_device();
    tile.set_fixed(fix_point);
    tile.take_step<clipped_spring>(0.1);
    tile.copy_to_host();

    MU_ASSERT("Fixed point moved in x", isclose(tile.h_X[fix_point].x, 0));
    MU_ASSERT("Fixed point moved in y", isclose(tile.h_X[fix_point].y, 0));
    MU_ASSERT("Fixed point moved in z", isclose(tile.h_X[fix_point].z, 0));

    return NULL;
}


__global__ void single_grid(
    const Grid* __restrict__ d_grid, const int grid_size)
{
    auto i = threadIdx.x + blockDim.x * threadIdx.y +
             blockDim.x * blockDim.y * threadIdx.z;

    auto cube_id_origin = (grid_size * grid_size * grid_size) / 2 +
                          (grid_size * grid_size) / 2 + grid_size / 2;
    auto expected_cube = cube_id_origin + threadIdx.x +
                         (grid_size * threadIdx.y) +
                         (grid_size * grid_size * threadIdx.z);

    auto one_point_per_cube = d_grid->d_cube_start[expected_cube] ==
                              d_grid->d_cube_end[expected_cube];
    D_ASSERT(one_point_per_cube);  // Thus no sorting!

    D_ASSERT(d_grid->d_cube_id[i] == expected_cube);
}

__global__ void double_grid(
    const Grid* __restrict__ d_grid, const int grid_size)
{
    auto i = threadIdx.x + blockDim.x * threadIdx.y +
             blockDim.x * blockDim.y * threadIdx.z;

    auto cube_id_origin = (grid_size * grid_size * grid_size) / 2 +
                          (grid_size * grid_size) / 2 + grid_size / 2;
    auto expected_cube =
        static_cast<int>(cube_id_origin + floor(threadIdx.x / 2.f) +
                         (grid_size * floor(threadIdx.y / 2.f)) +
                         (grid_size * grid_size * floor(threadIdx.z / 2.f)));

    auto in_expected_cube = false;
    for (auto j = d_grid->d_cube_start[expected_cube];
         j <= d_grid->d_cube_end[expected_cube]; j++) {
        if (d_grid->d_point_id[j] == i) in_expected_cube = true;
    }
    D_ASSERT(in_expected_cube);
}

const char* test_grid_spacing()
{
    const auto n_x = 7;
    const auto n_y = 7;
    const auto n_z = 7;

    Solution<float3, Grid_solver> points{n_x * n_y * n_z};
    for (auto i = 0; i < n_z; i++) {
        for (auto j = 0; j < n_y; j++) {
            for (auto k = 0; k < n_x; k++) {
                points.h_X[n_x * n_y * i + n_x * j + k].x = k + 0.5;
                points.h_X[n_x * n_y * i + n_x * j + k].y = j + 0.5;
                points.h_X[n_x * n_y * i + n_x * j + k].z = i + 0.5;
            }
        }
    }
    points.copy_to_device();

    auto grid_size = 70;
    Grid grid{n_x * n_y * n_z, grid_size};
    grid.build(points, 1);
    single_grid<<<1, dim3{n_x, n_y, n_z}>>>(grid.d_grid, grid_size);

    grid.build(points, 2);
    double_grid<<<1, dim3{n_x, n_y, n_z}>>>(grid.d_grid, grid_size);
    cudaDeviceSynchronize();  // Wait for device to exit

    return NULL;
}


const char* test_cube_size()
{
    Solution<float3, Grid_solver> points{2};
    points.h_X[0] = float3{0};
    points.h_X[1] = float3{0.75, 0, 0};
    points.copy_to_device();

    points.cube_size = 0.5;
    points.take_step<clipped_spring>(0.1);
    points.copy_to_host();
    MU_ASSERT("Cell outside cube moved", points.h_X[0].x == 0);

    points.cube_size = 1;
    points.take_step<clipped_spring>(0.1);
    points.copy_to_host();
    MU_ASSERT("Cell inside cube did not move", points.h_X[0].x != 0);

    return NULL;
}


__device__ int* d_n_nbs;

__device__ float3 count_neighbours(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > 1.0f) return dF;

    d_n_nbs[i] += 1;

    return dF;
}

const char* test_gabriel_solver()
{
    Solution<float3, Gabriel_solver> points{19, 5, 1.0f, 0.8f};
    *points.h_n = 19;
    regular_hexagon(0.5f, points);
    points.copy_to_device();

    Property<int> n_nbs{19};
    cudaMemcpyToSymbol(d_n_nbs, &n_nbs.d_prop, sizeof(d_n_nbs));
    auto reset_nbs = [&](const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, n_nbs.d_prop,
            n_nbs.d_prop + 19, 0);
    };

    points.take_step<count_neighbours>(0.1f, reset_nbs);
    n_nbs.copy_to_host();

    for(auto i = 0 ; i < 7 ; i++)
        MU_ASSERT("Wrong number of neighbours", n_nbs.h_prop[i] == 6);
    for(auto i = 7 ; i < 19 ; i++){
        if(i%2 != 0)
            MU_ASSERT("Wrong number of neighbours", n_nbs.h_prop[i] == 3);
        else
            MU_ASSERT("Wrong number of neighbours", n_nbs.h_prop[i] == 4);
    }

    return NULL;
}

const char* all_tests()
{
    MU_RUN_TEST(test_oscillation);
    MU_RUN_TEST(test_tile_tetrahedron);
    MU_RUN_TEST(test_grid_tetrahedron);
    MU_RUN_TEST(test_compare_methods);
    MU_RUN_TEST(test_generic_forces);
    MU_RUN_TEST(test_friction);
    MU_RUN_TEST(test_fix_point);
    MU_RUN_TEST(test_grid_spacing);
    MU_RUN_TEST(test_cube_size);
    MU_RUN_TEST(test_gabriel_solver);
    return NULL;
}

MU_RUN_SUITE(all_tests);
