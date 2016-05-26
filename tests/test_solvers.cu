#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "minunit.cuh"


const int N_MAX = 1000;
const float L_0 = 1;

__device__ __managed__ Solution<float3, N_MAX, N2nSolver> n2n;
__device__ __managed__ Solution<float3, N_MAX, LatticeSolver> latt;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    auto dF = float3{0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF = r*(L_0 - dist)/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ auto d_spring = spring;

const char* test_n2n_tetrahedron() {
    uniform_sphere(1, n2n, 4);
    auto com_i = center_of_mass(n2n, 4);
    for (auto i = 0; i < 500; i++) {
        n2n.step(0.1, d_spring, 4);
    }

    for (auto i = 1; i < 4; i++) {
        auto r = n2n[0] - n2n[i];
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in n2n tetrahedron", MU_ISCLOSE(dist, 1));
    }

    auto com_f = center_of_mass(n2n, 4);
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}

const char* test_latt_tetrahedron() {
    uniform_sphere(1, latt, 4);
    float3 com_i = center_of_mass(latt, 4);
    for (auto i = 0; i < 500; i++) {
        latt.step(0.1, d_spring, 4);
    }

    for (auto i = 1; i < 4; i++) {
        auto r = float3{latt[0].x - latt[i].x, latt[0].y - latt[i].y,
            latt[0].z - latt[i].z};
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in lattice tetrahedron", MU_ISCLOSE(dist, 1));
    }

    auto com_f = center_of_mass(latt, 4);
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    auto dF = float3{0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), 1);
    auto F = 2*(0.6 - dist)*(1 - dist) + (1 - dist)*(1 - dist);
    dF = r*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ auto d_cubic = clipped_cubic;

const char* test_compare_methods() {
    uniform_sphere(0.733333, n2n);
    for (auto i = 0; i < N_MAX; i++) {
        latt[i].x = n2n[i].x;
        latt[i].y = n2n[i].y;
        latt[i].z = n2n[i].z;
    }
    n2n.step(0.5, d_cubic);
    latt.step(0.5, d_cubic);

    for (auto i = 0; i < N_MAX; i++) {
        MU_ASSERT("Methods disagree", MU_ISCLOSE(latt[i].x, n2n[i].x));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(latt[i].y, n2n[i].y));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(latt[i].z, n2n[i].z));
    }

    return NULL;
}


const char* test_lattice_spacing() {
    for (auto i = 0; i < 10; i++) {
        for (auto j = 0; j < 10; j++) {
            for (auto k = 0; k < 10; k++) {
                latt[100*i + 10*j + k].x = k + 0.5;
                latt[100*i + 10*j + k].y = j + 0.5;
                latt[100*i + 10*j + k].z = i + 0.5;
            }
        }
    }

    latt.build_lattice(1000, 1);
    for (auto i = 0; i < 1000; i++) {
        auto expected_cube = pow(LATTICE_SIZE, 3)/2 + pow(LATTICE_SIZE, 2)/2 + LATTICE_SIZE/2
            + i%10 + (i%100/10)*LATTICE_SIZE + (i/100)*LATTICE_SIZE*LATTICE_SIZE;
        MU_ASSERT("Single lattice", latt.cube_id[i] == expected_cube);
    }

    latt.build_lattice(1000, 2);
    for (auto i = 0; i < 1000 - 8; i++) {
        MU_ASSERT("Double lattice", latt.cube_id[i] == latt.cube_id[i - i%8]);
    }

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(test_n2n_tetrahedron);
    MU_RUN_TEST(test_latt_tetrahedron);
    MU_RUN_TEST(test_compare_methods);
    MU_RUN_TEST(test_lattice_spacing);
    return NULL;
}

MU_RUN_SUITE(all_tests);
