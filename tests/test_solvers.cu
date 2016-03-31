#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "minunit.cuh"


const int N_MAX = 1000;
const float L_0 = 1;

__device__ __managed__ Solution<float3, N_MAX, N2nSolver> n2n;
__device__ __managed__ Solution<float3, N_MAX, LatticeSolver> latt;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF.x = r.x*(L_0 - dist)/dist;
    dF.y = r.y*(L_0 - dist)/dist;
    dF.z = r.z*(L_0 - dist)/dist;
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_spring = spring;

const char* test_n2n_tetrahedron() {
    uniform_sphere(1, n2n, 4);
    float3 com_i = center_of_mass(4, n2n);
    for (int i = 0; i < 500; i++) {
        n2n.step(0.1, p_spring, 4);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {n2n[0].x - n2n[i].x, n2n[0].y - n2n[i].y,
            n2n[0].z - n2n[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in n2n tetrahedron", mu_isclose(dist, 1));
    }
    float3 com_f = center_of_mass(4, n2n);
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.x, com_f.x));
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.y, com_f.y));
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.z, com_f.z));
    return NULL;
}

const char* test_latt_tetrahedron() {
    uniform_sphere(1, latt, 4);
    float3 com_i = center_of_mass(4, latt);
    for (int i = 0; i < 500; i++) {
        latt.step(0.1, p_spring, 4);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {latt[0].x - latt[i].x, latt[0].y - latt[i].y,
            latt[0].z - latt[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in lattice tetrahedron", mu_isclose(dist, 1));
    }
    float3 com_f = center_of_mass(4, latt);
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.x, com_f.x));
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.y, com_f.y));
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.z, com_f.z));
    return NULL;
}


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), 1);
    float F = 2*(0.6 - dist)*(1 - dist) + (1 - dist)*(1 - dist);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_cubic = clipped_cubic;

const char* test_compare_methods() {
    uniform_sphere(0.733333, n2n);
    for (int i = 0; i < N_MAX; i++) {
        latt[i].x = n2n[i].x;
        latt[i].y = n2n[i].y;
        latt[i].z = n2n[i].z;
    }
    n2n.step(0.5, p_cubic);
    latt.step(0.5, p_cubic);
    for (int i = 0; i < N_MAX; i++) {
        mu_assert("ERROR: Methods disagree", mu_isclose(latt[i].x, n2n[i].x));
        mu_assert("ERROR: Methods disagree", mu_isclose(latt[i].y, n2n[i].y));
        mu_assert("ERROR: Methods disagree", mu_isclose(latt[i].z, n2n[i].z));
    }
    return NULL;
}


const char* test_lattice_spacing() {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                latt[100*i + 10*j + k].x = k + 0.5;
                latt[100*i + 10*j + k].y = j + 0.5;
                latt[100*i + 10*j + k].z = i + 0.5;
            }
        }
    }
    latt.build_lattice(1000, 1);
    for (int i = 0; i < 1000; i++) {
        int expected_cube = pow(LATTICE_SIZE, 3)/2 + pow(LATTICE_SIZE, 2)/2 + LATTICE_SIZE/2
            + i%10 + (i%100/10)*LATTICE_SIZE + (i/100)*LATTICE_SIZE*LATTICE_SIZE;
        mu_assert("ERROR: Single lattice", latt.cube_id[i] == expected_cube);
    }
    latt.build_lattice(1000, 2);
    for (int i = 0; i < 1000 - 8; i++) {
        mu_assert("ERROR: Double lattice", latt.cube_id[i] == latt.cube_id[i - i%8]);
    }
    return NULL;
}


const char* all_tests() {
    mu_run_test(test_n2n_tetrahedron);
    mu_run_test(test_latt_tetrahedron);
    mu_run_test(test_compare_methods);
    mu_run_test(test_lattice_spacing);
    return NULL;
}

mu_run_suite(all_tests)
