#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/mesh.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


const char* test_transformations()
{
    Mesh mesh{"tests/torus.vtk"};
    auto minimum = mesh.get_minimum();
    auto maximum = mesh.get_maximum();
    MU_ASSERT("Min wrong in x", isclose(minimum.x, -1.5));
    MU_ASSERT("Min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Min wrong in z", isclose(minimum.z, -0.5));
    MU_ASSERT("Max wrong in x", isclose(maximum.x, 1.5));
    MU_ASSERT("Max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Max wrong in z", isclose(maximum.z, 0.5));

    mesh.translate(float3{1, 0, 0});
    minimum = mesh.get_minimum();
    maximum = mesh.get_maximum();
    MU_ASSERT("Translated min wrong in x", isclose(minimum.x, -1.5 + 1));
    MU_ASSERT("Translated min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Translated min wrong in z", isclose(minimum.z, -0.5));
    MU_ASSERT("Translated max wrong in x", isclose(maximum.x, 1.5 + 1));
    MU_ASSERT("Translated max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Translated max wrong in z", isclose(maximum.z, 0.5));
    mesh.translate(float3{-1, 0, 0});

    mesh.rotate(0, M_PI / 2, 0);
    minimum = mesh.get_minimum();
    maximum = mesh.get_maximum();
    MU_ASSERT("Rotated min wrong in x", isclose(minimum.x, -0.5));
    MU_ASSERT("Rotated min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Rotated min wrong in z", isclose(minimum.z, -1.5));
    MU_ASSERT("Rotated max wrong in x", isclose(maximum.x, 0.5));
    MU_ASSERT("Rotated max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Rotated max wrong in z", isclose(maximum.z, 1.5));
    mesh.rotate(0, -M_PI / 2, 0);

    mesh.rescale(2);
    minimum = mesh.get_minimum();
    maximum = mesh.get_maximum();
    MU_ASSERT("Scaled min wrong in x", isclose(minimum.x, -1.5 * 2));
    MU_ASSERT("Scaled min wrong in y", isclose(minimum.y, -1.5 * 2));
    MU_ASSERT("Scaled min wrong in z", isclose(minimum.z, -0.5 * 2));
    MU_ASSERT("Scaled max wrong in x", isclose(maximum.x, 1.5 * 2));
    MU_ASSERT("Scaled max wrong in y", isclose(maximum.y, 1.5 * 2));
    MU_ASSERT("Scaled max wrong in z", isclose(maximum.z, 0.5 * 2));
    mesh.rescale(0.5);

    mesh.grow_normally(0.1);
    minimum = mesh.get_minimum();
    maximum = mesh.get_maximum();
    MU_ASSERT("Grown min wrong in x", isclose(minimum.x, -1.5 - 0.1));
    MU_ASSERT("Grown min wrong in y", isclose(minimum.y, -1.5 - 0.1));
    MU_ASSERT("Grown min wrong in z", isclose(minimum.z, -0.5 - 0.1));
    MU_ASSERT("Grown max wrong in x", isclose(maximum.x, 1.5 + 0.1));
    MU_ASSERT("Grown max wrong in y", isclose(maximum.y, 1.5 + 0.1));
    MU_ASSERT("Grown max wrong in z", isclose(maximum.z, 0.5 + 0.1));

    return NULL;
}


const char* test_exclusion()
{
    const auto n_points = 1500;
    Solution<float3, Grid_solver> points{n_points};
    random_cuboid(
        0.25, float3{-1.5, -1.5, -0.5}, float3{1.5, 1.5, 0.5}, points);

    Mesh mesh{"tests/torus.vtk"};
    for (auto i = 0; i < n_points; i++) {
        auto dist_from_ring = sqrt(
            pow(1 - sqrt(pow(points.h_X[i].x, 2) + pow(points.h_X[i].y, 2)),
                2) +
            pow(points.h_X[i].z, 2));
        if (abs(dist_from_ring - 0.5) < 0.01) continue;  // Tolerance for mesh

        auto out = mesh.test_exclusion(points.h_X[i]);
        MU_ASSERT("Exclusion test wrong", (dist_from_ring >= 0.5) == out);
    }

    return NULL;
}


const char* test_shape_comparison()
{
    Mesh mesh{"tests/torus.vtk"};
    mesh.copy_to_device();
    Solution<float3, Grid_solver> points{
        static_cast<int>(mesh.vertices.size())};
    for (auto i = 0; i < mesh.vertices.size(); i++) {
        points.h_X[i].x = mesh.vertices[i].x;
        points.h_X[i].y = mesh.vertices[i].y;
        points.h_X[i].z = mesh.vertices[i].z;
    }
    points.copy_to_device();

    MU_ASSERT("Shape comparison wrong",
        isclose(mesh.shape_comparison_mesh_to_points(points), 0.0));

    mesh.grow_normally(0.1);
    mesh.copy_to_device();
    MU_ASSERT("Grown shape comparison wrong",
        isclose(mesh.shape_comparison_mesh_to_points(points), 0.1));

    return NULL;
}


__device__ __host__ bool operator==(const float3& a, const float3& b)
{
    return (a.x == b.x and a.y == b.y and a.z == b.z);
}

__device__ __host__ bool operator==(const Triangle& a, const Triangle& b)
{
    return (a.V0 == b.V0 and a.V1 == b.V1 and a.V2 == b.V2 and a.C == b.C and
            a.n == b.n);
}

const char* test_copy()
{
    Mesh orig{"tests/torus.vtk"};
    Mesh copy{orig};

    MU_ASSERT("Different vertices in copy", orig.vertices == copy.vertices);
    MU_ASSERT("Different facets in copy", orig.facets == copy.facets);
    MU_ASSERT("Different triangle_to_vertices in copy",
        orig.triangle_to_vertices == copy.triangle_to_vertices);
    MU_ASSERT("Different vertex_to_triangles in copy",
        orig.vertex_to_triangles == copy.vertex_to_triangles);

    copy.vertices.clear();
    MU_ASSERT("Removed vertices still in copy", orig.vertices != copy.vertices);

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_transformations);
    MU_RUN_TEST(test_exclusion);
    MU_RUN_TEST(test_shape_comparison);
    MU_RUN_TEST(test_copy);
    return NULL;
}

MU_RUN_SUITE(all_tests);
