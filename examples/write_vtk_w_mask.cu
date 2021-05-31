#include "../include/dtypes.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

MAKE_PT(Po_cell4, w, theta, phi);

int main()
{
    // Test writing & reading Solution
    const auto n_cells = 100;
    Solution<Po_cell4, Tile_solver> points_to_write{n_cells};

    bool mask[n_cells];
    for (auto i = 0; i < n_cells; i++) {
        points_to_write.h_X[i].x = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].y = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].z = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].w = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].phi = rand() / (RAND_MAX + 1.) * 2 * M_PI - M_PI;
        points_to_write.h_X[i].theta = acos(2. * rand() / (RAND_MAX + 1.) - 1);
        if(points_to_write.h_X[i].x>0.5f)
            mask[i] = true;
        else
            mask[i] = false;
    }


    Vtk_output output{"test_vtk", "output/", false};
    output.write_positions(points_to_write, mask);
    output.write_field(points_to_write, "w", &Po_cell4::w);


    return 0;
}
