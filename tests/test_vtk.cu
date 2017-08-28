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
    Solution<Po_cell4, n_cells, Tile_solver> bolls_to_write;
    Solution<Po_cell4, n_cells, Tile_solver> bolls_to_read;

    for (auto i = 0; i < n_cells; i++) {
        bolls_to_write.h_X[i].x = rand() / (RAND_MAX + 1.);
        bolls_to_write.h_X[i].y = rand() / (RAND_MAX + 1.);
        bolls_to_write.h_X[i].z = rand() / (RAND_MAX + 1.);
        bolls_to_write.h_X[i].w = rand() / (RAND_MAX + 1.);
        bolls_to_write.h_X[i].phi = rand() / (RAND_MAX + 1.) * 2 * M_PI - M_PI;
        bolls_to_write.h_X[i].theta = acos(2. * rand() / (RAND_MAX + 1.) - 1);
    }

    Vtk_output output("test_vtk");
    output.write_positions(bolls_to_write);
    output.write_polarity(bolls_to_write);
    output.write_field(bolls_to_write, "w", &Po_cell4::w);
    Vtk_input input("output/test_vtk_1.vtk");
    input.read_field(bolls_to_read, "w", &Po_cell4::w);
    input.read_polarity(bolls_to_read);
    input.read_positions(bolls_to_read);

    for (auto i = 0; i < n_cells; i++) {
        MU_ASSERT("Not close in x",
            MU_ISCLOSE(bolls_to_write.h_X[i].x, bolls_to_read.h_X[i].x));
        MU_ASSERT("Not close in y",
            MU_ISCLOSE(bolls_to_write.h_X[i].y, bolls_to_read.h_X[i].y));
        MU_ASSERT("Not close in z",
            MU_ISCLOSE(bolls_to_write.h_X[i].z, bolls_to_read.h_X[i].z));
        MU_ASSERT("Not close in w",
            MU_ISCLOSE(bolls_to_write.h_X[i].w, bolls_to_read.h_X[i].w));
        MU_ASSERT("Not close in phi",
            MU_ISCLOSE(bolls_to_write.h_X[i].phi, bolls_to_read.h_X[i].phi));
        MU_ASSERT("Not close in theta", MU_ISCLOSE(bolls_to_write.h_X[i].theta,
                                            bolls_to_read.h_X[i].theta));
    }

    // Test writing & reading Property
    Property<n_cells, int> ints_to_write("intprop");
    Property<n_cells, int> ints_to_read;
    Property<n_cells, float> floats_to_write("floatprop");
    Property<n_cells, float> floats_to_read;

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
            MU_ISCLOSE(ints_to_write.h_prop[i], ints_to_read.h_prop[i]));
        MU_ASSERT("Float property",
            MU_ISCLOSE(floats_to_write.h_prop[i], floats_to_read.h_prop[i]));
    }

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_io);
    printf("\n");
    return NULL;
}

MU_RUN_SUITE(all_tests);
