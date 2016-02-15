#include "minunit.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"


const int N_MAX_CELLS = 1000;
const float L_0 = 1;

__device__ __managed__ float3 n2n[N_MAX_CELLS];
__device__ __managed__ N2nSolver<float3, N_MAX_CELLS> n2n_solver;

__device__ __managed__ float3 latt[N_MAX_CELLS];
__device__ __managed__ LatticeSolver<float3, N_MAX_CELLS> latt_solver;


__device__ float3 neighbourhood_interaction(float3 Xi, float3 Xj, int i, int j) {
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


void global_interactions(const float3* __restrict__ X, float3* dX) {}


const char* test_n2n_tetrahedron() {
    uniform_sphere(4, 1, n2n);
    for (int i = 0; i < 500; i++) {
        n2n_solver.step(0.1, 4, n2n);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {n2n[0].x - n2n[i].x, n2n[0].y - n2n[i].y, n2n[0].z - n2n[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in n2n tetrahedron", mu_isclose(dist, 1));
    }
    return NULL;
}

const char* test_latt_tetrahedron() {
    uniform_sphere(4, 1, latt);
    for (int i = 0; i < 500; i++) {
        latt_solver.step(0.1, 4, latt);
    }
    for (int i = 1; i < 4; i++) {
        float3 r = {latt[0].x - latt[i].x, latt[0].y - latt[i].y, latt[0].z - latt[i].z};
        float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        mu_assert("ERROR: Spring not relaxed in lattice tetrahedron", mu_isclose(dist, 1));
    }
    return NULL;
}


const char* all_tests() {
    mu_run_test(test_n2n_tetrahedron);
    mu_run_test(test_latt_tetrahedron);
    return NULL;
}

mu_run_suite(all_tests)
