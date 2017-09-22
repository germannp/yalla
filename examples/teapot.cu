// Cut Utah teapot out of cuboid full of bolls
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <iterator>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/meix.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto n_cells = 1500u;


int main(int argc, const char* argv[])
{
    // Prepare cuboid
    Solution<float3, n_cells, Tile_solver> bolls;
    Meix teapot("tests/torus.vtk");
    random_cuboid(0.25, teapot.get_minimum(), teapot.get_maximum(), bolls);
    Vtk_output output("teapot");
    output.write_positions(bolls);

    // Cut teapot out
    auto new_n =
        thrust::remove_if(thrust::host, bolls.h_X, bolls.h_X + *bolls.h_n,
            [&teapot](float3 x) { return teapot.test_exclusion(x); });
    *bolls.h_n = std::distance(bolls.h_X, new_n);
    output.write_positions(bolls);

    return 0;
}
