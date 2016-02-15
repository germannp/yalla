#include "minunit.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"


const int N_MAX_CELLS = 1000;
const float L_0 = 1;

__device__ __managed__ N2nSolver<float3, N_MAX_CELLS> n2n;
__device__ __managed__ LatticeSolver<float3, N_MAX_CELLS> latt;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (i != j) {
        dF.x = r.x*(L_0 - dist)/dist;
        dF.y = r.y*(L_0 - dist)/dist;
        dF.z = r.z*(L_0 - dist)/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_spring = spring;

float3 center_of_mass(int N_CELLS, float3* X) {
    float3 com = {0, 0, 0};
    for (int i = 0; i < N_CELLS; i++) {
        com.x += X[i].x/N_CELLS;
        com.y += X[i].y/N_CELLS;
        com.z += X[i].z/N_CELLS;
    }
    return com;
}

const char* test_n2n_tetrahedron() {
    uniform_sphere(4, 1, n2n.X);
    float3 com_i = center_of_mass(4, n2n.X);
    for (int i = 0; i < 500; i++) {
        n2n.step(0.1, 4, p_spring);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {n2n.X[0].x - n2n.X[i].x, n2n.X[0].y - n2n.X[i].y,
            n2n.X[0].z - n2n.X[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in n2n tetrahedron", mu_isclose(dist, 1));
    }
    float3 com_f = center_of_mass(4, n2n.X);
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.x, com_f.x));
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.y, com_f.y));
    mu_assert("ERROR: Momentum in n2n tetrahedron", mu_isclose(com_i.z, com_f.z));
    return NULL;
}

const char* test_latt_tetrahedron() {
    uniform_sphere(4, 1, latt.X);
    float3 com_i = center_of_mass(4, latt.X);
    for (int i = 0; i < 500; i++) {
        latt.step(0.1, 4, p_spring);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {latt.X[0].x - latt.X[i].x, latt.X[0].y - latt.X[i].y,
            latt.X[0].z - latt.X[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in lattice tetrahedron", mu_isclose(dist, 1));
    }
    float3 com_f = center_of_mass(4, latt.X);
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.x, com_f.x));
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.y, com_f.y));
    mu_assert("ERROR: Momentum in lattice tetrahedron", mu_isclose(com_i.z, com_f.z));
    return NULL;
}


__device__ float3 clipped_cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF = {0.0f, 0.0f, 0.0f};
    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), 1);
    if (i != j) {
        float F = 2*(0.6 - dist)*(1 - dist) + (1 - dist)*(1 - dist);
        dF.x = r.x*F/dist;
        dF.y = r.y*F/dist;
        dF.z = r.z*F/dist;
    }
    assert(dF.x == dF.x); // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<float3> p_cubic = clipped_cubic;

const char* test_compare_methods() {
    uniform_sphere(N_MAX_CELLS, 0.733333, n2n.X);
    for (int i = 0; i < N_MAX_CELLS; i++) {
        latt.X[i].x = n2n.X[i].x;
        latt.X[i].y = n2n.X[i].y;
        latt.X[i].z = n2n.X[i].z;
    }
    n2n.step(0.5, N_MAX_CELLS, p_cubic);
    latt.step(0.5, N_MAX_CELLS, p_cubic);
    for (int i = 0; i < N_MAX_CELLS; i++) {
        mu_assert("ERROR: Methods disagree", mu_isclose(latt.X[i].x, n2n.X[i].x));
        mu_assert("ERROR: Methods disagree", mu_isclose(latt.X[i].y, n2n.X[i].y));
        mu_assert("ERROR: Methods disagree", mu_isclose(latt.X[i].z, n2n.X[i].z));
    }
    return NULL;
}


const char* all_tests() {
    mu_run_test(test_n2n_tetrahedron);
    mu_run_test(test_latt_tetrahedron);
    mu_run_test(test_compare_methods);
    return NULL;
}

mu_run_suite(all_tests)
