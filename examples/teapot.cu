// Cut Utah teapot out of cuboid full of points
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <iterator>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/mesh.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto n_points = 70000u;


int main(int argc, const char* argv[])
{
    // Prepare cuboid
    Solution<float3, Tile_solver> points{n_points};
    Mesh teapot{"examples/teapot.vtk"};
    random_cuboid(0.125, teapot.get_minimum(), teapot.get_maximum(), points);
    Vtk_output output{"teapot", "output/", false};
    output.write_positions(points);

    // Cut teapot out
    auto new_n =
        thrust::remove_if(thrust::host, points.h_X, points.h_X + *points.h_n,
            [&teapot](float3 x) { return teapot.test_exclusion(x); });
    *points.h_n = std::distance(points.h_X, new_n);
    output.write_positions(points);

    return 0;
}
