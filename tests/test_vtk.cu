#include "../include/dtypes.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include "minunit.cuh"

MAKE_PT(Po_cell4, w, theta, phi);

const char* test_io()
{
    // Test writing & reading Solution
    const auto n_cells = 100;
    Solution<Po_cell4, Tile_solver> points_to_write{n_cells};
    Solution<Po_cell4, Tile_solver> points_to_read{n_cells};

    for (auto i = 0; i < n_cells; i++) {
        points_to_write.h_X[i].x = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].y = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].z = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].w = rand() / (RAND_MAX + 1.);
        points_to_write.h_X[i].phi = rand() / (RAND_MAX + 1.) * 2 * M_PI - M_PI;
        points_to_write.h_X[i].theta = acos(2. * rand() / (RAND_MAX + 1.) - 1);
    }

    Vtk_output output{"test_vtk", "output/", false};
    output.write_positions(points_to_write);
    output.write_polarity(points_to_write);
    output.write_field(points_to_write, "w", &Po_cell4::w);
    Vtk_input input{"output/test_vtk_0.vtk"};
    input.read_field(points_to_read, "w", &Po_cell4::w);
    input.read_polarity(points_to_read);
    input.read_positions(points_to_read);

    for (auto i = 0; i < n_cells; i++) {
        MU_ASSERT("Not close in x",
            isclose(points_to_write.h_X[i].x, points_to_read.h_X[i].x));
        MU_ASSERT("Not close in y",
            isclose(points_to_write.h_X[i].y, points_to_read.h_X[i].y));
        MU_ASSERT("Not close in z",
            isclose(points_to_write.h_X[i].z, points_to_read.h_X[i].z));
        MU_ASSERT("Not close in w",
            isclose(points_to_write.h_X[i].w, points_to_read.h_X[i].w));
        MU_ASSERT("Not close in phi",
            isclose(points_to_write.h_X[i].phi, points_to_read.h_X[i].phi));
        MU_ASSERT("Not close in theta",
            isclose(points_to_write.h_X[i].theta, points_to_read.h_X[i].theta));
    }

    // Test writing & reading Property
    Property<int> ints_to_write{n_cells, "intprop"};
    Property<int> ints_to_read{n_cells};
    Property<float> floats_to_write{n_cells, "floatprop"};
    Property<float> floats_to_read{n_cells};

    for (auto i = 0; i < n_cells; i++) {
        ints_to_write.h_prop[i] = rand();
        floats_to_write.h_prop[i] = rand() / (RAND_MAX + 1.);
    }

    output.write_property(floats_to_write);
    output.write_property(ints_to_write);
    input.read_property(ints_to_read, "intprop");
    input.read_property(floats_to_read, "floatprop");

    for (auto i = 0; i < n_cells; i++) {
        MU_ASSERT("Int property",
            isclose(ints_to_write.h_prop[i], ints_to_read.h_prop[i]));
        MU_ASSERT("Float property",
            isclose(floats_to_write.h_prop[i], floats_to_read.h_prop[i]));
    }

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_io);
    return NULL;
}

MU_RUN_SUITE(all_tests);
